//! APIs for building a query of a database.

use std::{iter::once, sync::Arc};

use crate::{
    free_join::plan::{DecomposedPlan, JoinStageBlocks, SinglePlan},
    numeric_id::{DenseIdMap, IdVec, NumericId, define_id},
};
use smallvec::SmallVec;
use thiserror::Error;

use crate::{
    BaseValueId, CounterId, ExternalFunctionId, PoolSet, SortedWritesTable, Subset, SubsetRef,
    TaggedRowBuffer, Value,
    action::{Instr, QueryEntry, WriteVal},
    common::HashMap,
    free_join::{
        ActionId, AtomId, Database, ProcessedConstraints, SubAtom, TableId, TableInfo, VarInfo,
        Variable,
        plan::{JoinHeader, Plan, PlanStrategy},
    },
    pool::{Pooled, with_pool_set},
    table_spec::{ColumnId, Constraint, Offset},
};

define_id!(pub RuleId, u32, "An identifier for a rule in a rule set");

/// Resolves variables and atoms in a rule to their string names.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SymbolMap {
    pub atoms: HashMap<AtomId, Arc<str>>,
    pub vars: HashMap<Variable, Arc<str>>,
}

/// A cached plan for a given rule.
pub struct CachedPlan {
    plan: Plan,
    desc: Arc<str>,
    symbol_map: SymbolMap,
    actions: ActionInfo,
}

impl CachedPlan {
    /// The table that `atom` of this plan reads.
    pub fn atom_table(&self, atom: AtomId) -> TableId {
        self.plan.atoms_ref()[atom].table
    }

    /// Whether the plan restricts its atoms with constraints of its own, so
    /// that a variant can select no rows even when its extra constraints do.
    pub fn has_cached_constraints(&self) -> bool {
        !self.plan.header().is_empty()
    }

    /// The columns of `small` that bind the key of `keyed`, in key order, if
    /// `keyed` reads a [`SortedWritesTable`] (which has one live row per key)
    /// and `small` binds every key column with a variable.
    pub fn key_join(
        &self,
        db: &Database,
        small: AtomId,
        keyed: AtomId,
    ) -> Option<SmallVec<[ColumnId; 4]>> {
        let atoms = self.plan.atoms_ref();
        let (small, keyed) = (&atoms[small], &atoms[keyed]);
        let keyed_info = db.get_table_info(keyed.table);
        keyed_info
            .table
            .as_any()
            .downcast_ref::<SortedWritesTable>()?;
        (0..keyed_info.spec.n_keys)
            .map(|col| small.get_col(keyed.get_var(ColumnId::from_usize(col))?))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ActionInfo {
    pub(crate) used_vars: SmallVec<[Variable; 4]>,
    pub(crate) instrs: Arc<Pooled<Vec<Instr>>>,
}

/// A set of rules to run against a [`Database`].
///
/// See [`Database::new_rule_set`] for more information.
#[derive(Default)]
pub struct RuleSet {
    /// The contents of the queries (i.e. the LHS of the rules) for each rule in the set, along
    /// with a description of the rule.
    ///
    /// The action here is used to map between rule descriptions and plans, which contain ActionIds. The current
    /// accounting logic assumes that rules and actions stand in a bijection. If we relaxed that
    /// later on, most of the core logic would still work but the accounting logic could get more
    /// complex.
    pub(crate) plans: IdVec<RuleId, (Plan, Arc<str> /* description */, SymbolMap)>,
    pub(crate) actions: DenseIdMap<ActionId, ActionInfo>,
    /// Descriptions of rules dropped because a key lookup showed they match
    /// nothing. Each is reported as a run with no matches, as if executed.
    pub(crate) unmatched: Vec<Arc<str>>,
    /// Whether a rule that would have run was left out, so running this set
    /// must still merge pending updates as running that rule would have.
    pub(crate) left_out: bool,
}

impl RuleSet {
    pub fn build_cached_plan(&self, rule_id: RuleId) -> CachedPlan {
        let (plan, desc, symbol_map) = self.plans.get(rule_id).expect("rule must exist");
        let actions = self
            .actions
            .get(plan.actions())
            .expect("action must exist")
            .clone();
        CachedPlan {
            plan: plan.clone(),
            desc: desc.clone(),
            symbol_map: symbol_map.clone(),
            actions,
        }
    }
}

/// Builder for a [`RuleSet`].
///
/// There are in general two ways to add rules to a rule set:
///
/// 1. Use the QueryBuilder and RuleBuilder APIs to construct a rule from scratch.
/// 2. Use a previously cached plan and add extra constraints to it.
///
/// The pattern this is used by egglog is as follows: An egglog rule is first compiled to a cached
/// plan using builder patterns at declaration time, and each time the rule is run, it is added to
/// a ruleset using the cached plan and possibly some extra constraints (e.g., timestamp).
///
/// See [`Database::new_rule_set`] for more information.
pub struct RuleSetBuilder<'outer> {
    rule_set: RuleSet,
    db: &'outer mut Database,
    unmatched: UnmatchedRules,
}

/// What [`RuleSetBuilder::add_rule_from_cached_plan`] does with a rule that a
/// key lookup shows matches nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnmatchedRules {
    /// Add it, and run it.
    #[default]
    Run,
    /// Leave it out, and report it as run with no matches and no plan.
    Report,
    /// Leave it out of the rule set and its report.
    Drop,
}

/// The most candidate rows [`RuleSetBuilder::key_lookup_rules_out`] scans in
/// one atom.
const SMALL_ATOM_ROWS: usize = 4;

fn satisfies(constraints: &[Constraint], row: &[Value]) -> bool {
    constraints.iter().all(|c| match c {
        Constraint::Eq { l_col, r_col } => row[l_col.index()] == row[r_col.index()],
        Constraint::EqConst { col, val } => row[col.index()] == *val,
        Constraint::LtConst { col, val } => row[col.index()] < *val,
        Constraint::GtConst { col, val } => row[col.index()] > *val,
        Constraint::LeConst { col, val } => row[col.index()] <= *val,
        Constraint::GeConst { col, val } => row[col.index()] >= *val,
    })
}

impl<'outer> RuleSetBuilder<'outer> {
    pub fn new(db: &'outer mut Database) -> Self {
        Self {
            rule_set: Default::default(),
            db,
            unmatched: UnmatchedRules::Run,
        }
    }

    /// The keys of the live rows of `atom`'s table in `cached`, flattened in
    /// the order of `key_cols`, or `None` if it has more than
    /// [`SMALL_ATOM_ROWS`] live rows.
    pub fn small_atom_keys(
        &self,
        cached: &CachedPlan,
        atom: AtomId,
        key_cols: &[ColumnId],
    ) -> Option<SmallVec<[Value; 4]>> {
        let info = self.db.get_table_info(cached.atom_table(atom));
        if info.table.len() > SMALL_ATOM_ROWS {
            return None;
        }
        let mut rows = TaggedRowBuffer::new(info.spec.arity());
        info.table.scan_bounded(
            info.table.all().as_ref(),
            Offset::new(0),
            usize::MAX,
            &mut rows,
        );
        Some(
            rows.non_stale()
                .flat_map(|(_, row)| key_cols.iter().map(|col| row[col.index()]))
                .collect(),
        )
    }

    /// Whether some key tuple in `keys` (key tuples of `table`, flattened)
    /// has a live row in `table` that meets `constraint`, or `None` if
    /// `constraint` compares two columns or `table` has no key columns.
    pub fn any_key_meets(
        &self,
        table: TableId,
        constraint: &Constraint,
        keys: &[Value],
    ) -> Option<bool> {
        let info = self.db.get_table_info(table);
        let n_keys = info.spec.n_keys;
        let (col, val, meets): (ColumnId, Value, fn(Value, Value) -> bool) = match *constraint {
            Constraint::Eq { .. } => return None,
            Constraint::EqConst { col, val } => (col, val, |x, c| x == c),
            Constraint::LtConst { col, val } => (col, val, |x, c| x < c),
            Constraint::GtConst { col, val } => (col, val, |x, c| x > c),
            Constraint::LeConst { col, val } => (col, val, |x, c| x <= c),
            Constraint::GeConst { col, val } => (col, val, |x, c| x >= c),
        };
        if n_keys == 0 {
            return None;
        }
        Some(keys.chunks(n_keys).any(|key| {
            info.table
                .get_row_column(key, col)
                .is_some_and(|x| meets(x, val))
        }))
    }

    /// Set what [`RuleSetBuilder::add_rule_from_cached_plan`] does with a rule
    /// that a key lookup shows matches nothing.
    pub fn unmatched_rules(&mut self, unmatched: UnmatchedRules) {
        self.unmatched = unmatched;
    }

    /// What [`RuleSetBuilder::add_rule_from_cached_plan`] does with a rule
    /// that a key lookup shows matches nothing.
    pub fn unmatched_rules_mode(&self) -> UnmatchedRules {
        self.unmatched
    }

    /// Leave out `cached`, which a key lookup shows matches nothing, as
    /// [`RuleSetBuilder::unmatched_rules`] says.
    pub fn leave_out(&mut self, cached: &CachedPlan) {
        self.rule_set.left_out = true;
        if self.unmatched == UnmatchedRules::Report {
            self.rule_set.unmatched.push(cached.desc.clone());
        }
    }

    /// Estimate the size of the subset of the table matching the given
    /// constraint.
    ///
    /// This is a wrapper around the [`Database::estimate_size`] method.
    pub fn estimate_size(&self, table: TableId, c: Option<Constraint>) -> usize {
        self.db.estimate_size(table, c)
    }

    /// Add a rule to this rule set.
    pub fn new_rule<'a>(&'a mut self) -> QueryBuilder<'outer, 'a> {
        let instrs = with_pool_set(PoolSet::get);
        QueryBuilder {
            rsb: self,
            instrs,
            query: Query {
                var_info: Default::default(),
                atoms: Default::default(),
                // start with an invalid ActionId
                action: ActionId::new(u32::MAX),
                plan_strategy: Default::default(),
                fun_deps: Default::default(),
                no_decomp: false,
            },
        }
    }

    fn reprocess_constraints(
        &self,
        table: TableId,
        atom: AtomId,
        constraints: &[Constraint],
    ) -> Option<JoinHeader> {
        let processed = self.db.process_constraints(table, constraints);
        if !processed.slow.is_empty() {
            panic!(
                "Cached plans only support constraints with a fast pushdown. \
                 Got: {constraints:?} for table {table:?}",
            );
        }
        if processed.subset.size() == 0 {
            return None;
        }
        Some(JoinHeader {
            atom,
            constraints: processed.fast,
            subset: processed.subset,
        })
    }

    fn push_extra_constraints(
        &self,
        headers: &mut Vec<JoinHeader>,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        extra_constraints: &[(AtomId, Constraint)],
    ) -> Option<()> {
        for (atom_id, constraint) in extra_constraints {
            let atom_info = atoms.get(*atom_id).expect("atom must exist in plan");
            let table = atom_info.table;
            headers.push(self.reprocess_constraints(
                table,
                *atom_id,
                std::slice::from_ref(constraint),
            )?);
        }
        Some(())
    }

    fn reprocess_existing_headers(
        &self,
        headers: &mut Vec<JoinHeader>,
        atoms: &Arc<DenseIdMap<AtomId, Atom>>,
        existing: &[JoinHeader],
    ) -> Option<()> {
        for JoinHeader {
            atom, constraints, ..
        } in existing
        {
            let atom_info = atoms.get(*atom).expect("atom must exist in plan");
            let table = atom_info.table;
            headers.push(self.reprocess_constraints(table, *atom, constraints)?);
        }
        Some(())
    }

    fn get_rule_with_extra_constraints(
        &self,
        cached: &CachedPlan,
        action_id: ActionId,
        extra_constraints: &[(AtomId, Constraint)],
    ) -> Option<Plan> {
        match &cached.plan {
            Plan::SinglePlan(cached_plan) => {
                let mut headers = vec![];
                let stages = cached_plan.stages.clone();
                self.push_extra_constraints(&mut headers, &cached_plan.atoms, extra_constraints)?;
                self.reprocess_existing_headers(
                    &mut headers,
                    &cached_plan.atoms,
                    &cached_plan.header,
                )?;
                Some(Plan::SinglePlan(SinglePlan {
                    atoms: cached_plan.atoms.clone(),
                    header: headers,
                    stages,
                    actions: action_id,
                }))
            }
            Plan::DecomposedPlan(cached_plan) => {
                let mut blocks = Vec::with_capacity(cached_plan.stages.blocks.len());
                let mut headers = vec![];
                self.push_extra_constraints(&mut headers, &cached_plan.atoms, extra_constraints)?;
                self.reprocess_existing_headers(
                    &mut headers,
                    &cached_plan.atoms,
                    &cached_plan.header,
                )?;
                for cached_block in cached_plan.stages.blocks.iter() {
                    let stages = cached_block.0.clone();
                    blocks.push((stages, cached_block.1.clone()));
                }
                let result_block = cached_plan.result_block.clone();
                Some(Plan::DecomposedPlan(DecomposedPlan {
                    atoms: cached_plan.atoms.clone(),
                    header: headers,
                    stages: JoinStageBlocks { blocks },
                    actions: action_id,
                    result_block,
                }))
            }
        }
    }

    /// Whether a key lookup from a small atom shows that `atoms` have no match
    /// under `restrictions`, each an atom with a superset of the rows it may
    /// match and constraints those rows meet.
    ///
    /// Take an atom with at most [`SMALL_ATOM_ROWS`] candidate rows and a
    /// restricted atom over a [`SortedWritesTable`] whose key columns hold
    /// variables the small atom binds. Every match pairs a live row of the
    /// small atom with the one live row at the key it binds, and that row
    /// meets the keyed atom's constraints. If no small row finds such a row,
    /// nothing matches. A term-encoding rebuild rule for a one-row view joins
    /// it with the few union-find rows written since its last run; this
    /// rejects it without running the join.
    fn key_lookup_rules_out(
        &self,
        atoms: &DenseIdMap<AtomId, Atom>,
        restrictions: &[(AtomId, SubsetRef<'_>, &[Constraint])],
    ) -> bool {
        for (small_id, small) in atoms.iter() {
            let small_info = self.db.get_table_info(small.table);
            let small_restrictions = || restrictions.iter().filter(move |r| r.0 == small_id);
            let all;
            let candidates = match small_restrictions().min_by_key(|r| r.1.size()) {
                Some((_, subset, _)) => *subset,
                None => {
                    all = small_info.table.all();
                    all.as_ref()
                }
            };
            if candidates.size() > SMALL_ATOM_ROWS {
                continue;
            }
            let mut rows = None;
            for (keyed_id, keyed) in atoms.iter() {
                let keyed_restrictions = || restrictions.iter().filter(move |r| r.0 == keyed_id);
                if keyed_id == small_id || keyed_restrictions().next().is_none() {
                    continue;
                }
                let keyed_info = self.db.get_table_info(keyed.table);
                if keyed_info
                    .table
                    .as_any()
                    .downcast_ref::<SortedWritesTable>()
                    .is_none()
                {
                    continue;
                }
                let Some(key_cols) = (0..keyed_info.spec.n_keys)
                    .map(|col| small.get_col(keyed.get_var(ColumnId::from_usize(col))?))
                    .collect::<Option<SmallVec<[ColumnId; 4]>>>()
                else {
                    continue;
                };
                let rows = rows.get_or_insert_with(|| {
                    let mut rows = TaggedRowBuffer::new(small_info.spec.arity());
                    small_info.table.scan_bounded(
                        candidates,
                        Offset::new(0),
                        usize::MAX,
                        &mut rows,
                    );
                    rows
                });
                let mut key = SmallVec::<[Value; 4]>::new();
                let may_match = rows.non_stale().any(|(_, row)| {
                    if !small_restrictions().all(|r| satisfies(r.2, row)) {
                        return false;
                    }
                    key.clear();
                    key.extend(key_cols.iter().map(|col| row[col.index()]));
                    keyed_info.table.get_row(&key).is_some_and(|found| {
                        keyed_restrictions().all(|r| satisfies(r.2, &found.vals))
                    })
                });
                if !may_match {
                    return true;
                }
            }
        }
        false
    }

    /// The rows of `table` that `constraint` selects, computed as
    /// [`Database::process_constraints`] does, if it is a fast constraint.
    fn fast_subset(&self, table: TableId, constraint: &Constraint) -> Option<Subset> {
        let table = self.db.get_table(table);
        let fast = table.fast_subset(constraint)?;
        let mut subset = table.all();
        with_pool_set(|ps| subset.intersect(fast.as_ref(), &ps.get_pool()));
        Some(subset)
    }

    /// Whether to drop `cached` under `extra_constraints` because a key
    /// lookup shows it matches nothing, decided before building its headers.
    ///
    /// Only plans without cached headers qualify: their variant is empty
    /// exactly when an extra constraint selects no rows, so a dropped rule is
    /// one that would otherwise have run.
    fn drop_before_headers(
        &self,
        cached: &CachedPlan,
        extra_constraints: &[(AtomId, Constraint)],
    ) -> bool {
        if self.unmatched == UnmatchedRules::Run || !cached.plan.header().is_empty() {
            return false;
        }
        let mut subsets = SmallVec::<[(AtomId, Subset, &Constraint); 4]>::new();
        for (atom, constraint) in extra_constraints {
            let Some(subset) = self.fast_subset(cached.atom_table(*atom), constraint) else {
                return false;
            };
            if subset.size() == 0 {
                return false;
            }
            subsets.push((*atom, subset, constraint));
        }
        let restrictions: SmallVec<[_; 4]> = subsets
            .iter()
            .map(|(atom, subset, constraint)| {
                (*atom, subset.as_ref(), std::slice::from_ref(*constraint))
            })
            .collect();
        self.key_lookup_rules_out(cached.plan.atoms_ref(), &restrictions)
    }

    /// Whether the rows of `table` may have changed since
    /// [`Database::change_epoch`] returned `epoch`.
    pub fn changed_since(&self, table: TableId, epoch: u64) -> bool {
        self.db.changed_since(table, epoch)
    }

    /// The database's [`Database::change_epoch`].
    pub fn change_epoch(&self) -> u64 {
        self.db.change_epoch()
    }

    /// Whether `constraint` selects no rows of `table`.
    ///
    /// If so, [`RuleSetBuilder::add_rule_from_cached_plan`] returns `None` for
    /// any constraint list that puts it on an atom over `table`, so a caller
    /// can test this before assembling the rest of the list.
    pub fn constraint_is_empty(&self, table: TableId, constraint: &Constraint) -> bool {
        self.db
            .get_table(table)
            .fast_subset(constraint)
            .is_some_and(|subset| subset.size() == 0)
    }

    /// Add a rule to this rule set based on a previously cached plan, optionally
    /// with additional constraints applied on top.
    ///
    /// Returns `None` if the query is provably empty given the current database
    /// state (i.e. some constraint narrows a table to zero matching rows), in
    /// which case no rule or action is allocated. Returns `Some(RuleId)` otherwise.
    ///
    /// The primary use-case is seminaive evaluation: an egglog rule is compiled
    /// once into a [`CachedPlan`] and then added to a fresh [`RuleSet`] each
    /// iteration with timestamp constraints (e.g. `GeConst` on the focus atom)
    /// that select only new tuples. If no new tuples exist for an atom, the
    /// `None` return allows the caller to skip that variant entirely.
    pub fn add_rule_from_cached_plan(
        &mut self,
        cached: &CachedPlan,
        extra_constraints: &[(AtomId, Constraint)],
    ) -> Option<RuleId> {
        // Peek at the action_id without allocating it yet, so we don't break
        // the rules<->actions bijection if the query turns out to be empty.
        let action_id = self.rule_set.actions.next_id();
        if self.drop_before_headers(cached, extra_constraints) {
            self.leave_out(cached);
            return None;
        }
        let plan = self.get_rule_with_extra_constraints(cached, action_id, extra_constraints)?;
        if self.unmatched != UnmatchedRules::Run && !cached.plan.header().is_empty() {
            let restrictions: SmallVec<[_; 4]> = plan
                .header()
                .iter()
                .map(|h| (h.atom, h.subset.as_ref(), &h.constraints[..]))
                .collect();
            if self.key_lookup_rules_out(plan.atoms_ref(), &restrictions) {
                self.leave_out(cached);
                return None;
            }
        }
        // The query is non-empty: now commit the action and the plan.
        let actual_action_id = self.rule_set.actions.push(cached.actions.clone());
        debug_assert_eq!(action_id, actual_action_id);
        Some(
            self.rule_set
                .plans
                .push((plan, cached.desc.clone(), cached.symbol_map.clone())),
        )
    }

    /// Build the ruleset.
    pub fn build(self) -> RuleSet {
        self.rule_set
    }
}

/// Builder for the "query" portion of the rule.
///
/// Queries specify scans or joins over the database that bind variables that
/// are accessible to rules.
pub struct QueryBuilder<'outer, 'a> {
    rsb: &'a mut RuleSetBuilder<'outer>,
    query: Query,
    instrs: Pooled<Vec<Instr>>,
}

impl<'outer, 'a> QueryBuilder<'outer, 'a> {
    /// Finish the query and start building the right-hand side of the rule.
    pub fn build(self) -> RuleBuilder<'outer, 'a> {
        RuleBuilder { qb: self }
    }

    /// Set the target plan strategy to use to execute this query.
    pub fn set_plan_strategy(&mut self, strategy: PlanStrategy) {
        self.query.plan_strategy = strategy;
    }

    /// If `true`, the query planner will skip tree-decomposition
    /// for query decomposition and always use evaluate the query as a single bag.
    pub fn set_no_decomp(&mut self, no_decomp: bool) {
        self.query.no_decomp = no_decomp;
    }

    /// Create a new variable of the given type.
    pub fn new_var(&mut self) -> Variable {
        self.query.var_info.push(VarInfo {
            occurrences: Default::default(),
            used_in_rhs: false,
            defined_in_rhs: false,
            name: None,
        })
    }

    pub fn new_var_named(&mut self, name: &str) -> Variable {
        self.query.var_info.push(VarInfo {
            occurrences: Default::default(),
            used_in_rhs: false,
            defined_in_rhs: false,
            name: Some(name.into()),
        })
    }

    fn mark_used<'b>(&mut self, entries: impl IntoIterator<Item = &'b QueryEntry>) {
        for entry in entries {
            if let QueryEntry::Var(v) = entry {
                self.query.var_info[*v].used_in_rhs = true;
            }
        }
    }

    fn mark_defined(&mut self, entry: &QueryEntry) {
        // TODO: use some of this information in query planning, e.g. dedup at match time.
        if let QueryEntry::Var(v) = entry {
            self.query.var_info[*v].defined_in_rhs = true;
        }
    }

    /// Add the given atom to the query, with the given variables and constraints.
    ///
    /// NB: it is possible to constrain two non-equal variables to be equal
    /// given this setup. Doing this will not cause any problems but
    /// nevertheless is not recommended.
    ///
    /// The returned `AtomId` can be used to refer to this atom when adding constraints in
    /// [`RuleSetBuilder::add_rule_from_cached_plan`].
    ///
    /// # Panics
    /// Like most methods that take a [`TableId`], this method will panic if the
    /// given table is not declared in the corresponding database.
    pub fn add_atom<'b>(
        &mut self,
        table_id: TableId,
        vars: &[QueryEntry],
        cs: impl IntoIterator<Item = &'b Constraint>,
    ) -> Result<AtomId, QueryError> {
        let info = &self.rsb.db.tables[table_id];
        let arity = info.spec.arity();
        let check_constraint = |c: &Constraint| {
            let process_col = |col: &ColumnId| -> Result<(), QueryError> {
                if col.index() >= arity {
                    Err(QueryError::InvalidConstraint {
                        constraint: c.clone(),
                        column: col.index(),
                        table: table_id,
                        arity,
                    })
                } else {
                    Ok(())
                }
            };
            match c {
                Constraint::Eq { l_col, r_col } => {
                    process_col(l_col)?;
                    process_col(r_col)
                }
                Constraint::EqConst { col, .. }
                | Constraint::LtConst { col, .. }
                | Constraint::GtConst { col, .. }
                | Constraint::LeConst { col, .. }
                | Constraint::GeConst { col, .. } => process_col(col),
            }
        };
        if arity != vars.len() {
            return Err(QueryError::BadArity {
                table: table_id,
                expected: arity,
                got: vars.len(),
            });
        }
        let cs = Vec::from_iter(
            cs.into_iter()
                .cloned()
                .chain(vars.iter().enumerate().filter_map(|(i, qe)| match qe {
                    QueryEntry::Var(_) => None,
                    QueryEntry::Const(c) => Some(Constraint::EqConst {
                        col: ColumnId::from_usize(i),
                        val: *c,
                    }),
                })),
        );
        cs.iter().try_fold((), |_, c| check_constraint(c))?;
        let processed = self.rsb.db.process_constraints(table_id, &cs);
        let mut atom = Atom {
            table: table_id,
            var_columns: Default::default(),
            constraints: processed,
        };
        let next_atom = AtomId::from_usize(self.query.atoms.n_ids());
        let mut subatoms = HashMap::<Variable, SubAtom>::default();
        for (i, qe) in vars.iter().enumerate() {
            let var = match qe {
                QueryEntry::Var(var) => *var,
                QueryEntry::Const(_) => {
                    continue;
                }
            };
            if var == Variable::placeholder() {
                continue;
            }
            let col = ColumnId::from_usize(i);
            if let Some(prev) = atom.var_columns.insert(var, col) {
                atom.constraints.slow.push(Constraint::Eq {
                    l_col: col,
                    r_col: prev,
                })
            };
            subatoms
                .entry(var)
                .or_insert_with(|| SubAtom::new(next_atom))
                .vars
                .push(col);
        }
        for (var, subatom) in subatoms {
            self.query
                .var_info
                .get_mut(var)
                .expect("all variables must be bound in current query")
                .occurrences
                .push(subatom);
        }

        // Add functional dependencies for this atom.
        let get_var = |qe: &QueryEntry| match qe {
            QueryEntry::Var(v) => Some(*v),
            QueryEntry::Const(_) => None,
        };
        let antecedent = vars[..info.spec().n_keys]
            .iter()
            .filter_map(get_var)
            .collect::<Vec<_>>();
        let consequent = vars[info.spec().n_keys..]
            .iter()
            .filter_map(get_var)
            .collect::<Vec<_>>();
        self.query.fun_deps.add_dependency(antecedent, consequent);

        Ok(self.query.atoms.push(atom))
    }
}

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("table {table:?} has {expected:?} keys but got {got:?}")]
    KeyArityMismatch {
        table: TableId,
        expected: usize,
        got: usize,
    },
    #[error("table {table:?} has {expected:?} columns but got {got:?}")]
    TableArityMismatch {
        table: TableId,
        expected: usize,
        got: usize,
    },

    #[error(
        "counter used in column {column_id:?} of table {table:?}, which is declared as a base value"
    )]
    CounterUsedInBaseColumn {
        table: TableId,
        column_id: ColumnId,
        base: BaseValueId,
    },

    #[error("attempt to compare two groups of values, one of length {l}, another of length {r}")]
    MultiComparisonMismatch { l: usize, r: usize },

    #[error("table {table:?} expected {expected:?} columns but got {got:?}")]
    BadArity {
        table: TableId,
        expected: usize,
        got: usize,
    },

    #[error("expected {expected:?} columns in schema but got {got:?}")]
    InvalidSchema { expected: usize, got: usize },

    #[error(
        "constraint {constraint:?} on table {table:?} references column {column:?}, but the table has arity {arity:?}"
    )]
    InvalidConstraint {
        constraint: Constraint,
        column: usize,
        table: TableId,
        arity: usize,
    },
}

/// Builder for the "action" portion of the rule.
///
/// Rules can refer to the variables bound in their query to modify the database.
pub struct RuleBuilder<'outer, 'a> {
    qb: QueryBuilder<'outer, 'a>,
}

impl RuleBuilder<'_, '_> {
    fn table_info(&self, table: TableId) -> &TableInfo {
        self.qb.rsb.db.get_table_info(table)
    }

    /// Build the finished query.
    pub fn build(self) -> RuleId {
        self.build_with_description("")
    }

    fn build_symbol_map(&self) -> SymbolMap {
        let var_info = &self.qb.query.var_info;
        SymbolMap {
            atoms: self
                .qb
                .query
                .atoms
                .iter()
                .filter_map(|(id, atom)| {
                    let name = self.table_info(atom.table).name.clone();
                    name.map(|name| (id, name))
                })
                .collect(),
            vars: var_info
                .iter()
                .filter_map(|(id, info)| info.name.as_ref().map(|name| (id, name.clone())))
                .collect(),
        }
    }

    pub fn build_with_description(mut self, desc: impl Into<String>) -> RuleId {
        let var_info = &self.qb.query.var_info;
        let symbol_map = self.build_symbol_map();
        // Generate an id for our actions and slot them in.
        let used_vars = SmallVec::from_iter(var_info.iter().filter_map(|(v, info)| {
            if info.used_in_rhs && !info.defined_in_rhs {
                Some(v)
            } else {
                None
            }
        }));
        let action_id = self.qb.rsb.rule_set.actions.push(ActionInfo {
            instrs: Arc::new(self.qb.instrs),
            used_vars,
        });
        self.qb.query.action = action_id;
        // Plan the query
        let plan = self.qb.rsb.db.plan_query(self.qb.query);
        let desc: String = desc.into();
        // Add it to the ruleset.
        self.qb
            .rsb
            .rule_set
            .plans
            .push((plan, desc.into(), symbol_map))
    }

    /// Return a variable containing the result of reading the specified counter.
    pub fn read_counter(&mut self, counter: CounterId) -> Variable {
        let dst = self.qb.new_var();
        self.qb.instrs.push(Instr::ReadCounter { counter, dst });
        self.qb.mark_defined(&dst.into());
        dst
    }

    /// Return a variable containing the result of looking up the specified
    /// column from the row corresponding to given keys in the given
    /// table.
    ///
    /// If the key does not currently have a mapping in the table, the values
    /// specified by `default_vals` will be inserted.
    pub fn lookup_or_insert(
        &mut self,
        table: TableId,
        args: &[QueryEntry],
        default_vals: &[WriteVal],
        dst_col: ColumnId,
    ) -> Result<Variable, QueryError> {
        let table_info = self.table_info(table);
        self.validate_keys(table, table_info, args)?;
        self.validate_vals(table, table_info, default_vals.iter())?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::LookupOrInsertDefault {
            table,
            args: args.to_vec(),
            default: default_vals.to_vec(),
            dst_col,
            dst_var: res,
        });
        self.qb.mark_used(args);
        self.qb
            .mark_used(default_vals.iter().filter_map(|x| match x {
                WriteVal::QueryEntry(qe) => Some(qe),
                WriteVal::IncCounter(_) | WriteVal::CurrentVal(_) => None,
            }));
        self.qb.mark_defined(&res.into());
        Ok(res)
    }

    /// Return a variable containing the result of looking up the specified
    /// column from the row corresponding to given keys in the given
    /// table.
    ///
    /// If the key does not currently have a mapping in the table, the variable
    /// takes the value of `default`.
    pub fn lookup_with_default(
        &mut self,
        table: TableId,
        args: &[QueryEntry],
        default: QueryEntry,
        dst_col: ColumnId,
    ) -> Result<Variable, QueryError> {
        let table_info = self.table_info(table);
        self.validate_keys(table, table_info, args)?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::LookupWithDefault {
            table,
            args: args.to_vec(),
            dst_col,
            dst_var: res,
            default,
        });
        self.qb.mark_used(args);
        self.qb.mark_used(&[default]);
        self.qb.mark_defined(&res.into());
        Ok(res)
    }

    /// Return a variable containing the result of looking up the specified
    /// column from the row corresponding to given keys in the given
    /// table.
    ///
    /// If the key does not currently have a mapping in the table, execution of
    /// the rule is halted.
    pub fn lookup(
        &mut self,
        table: TableId,
        args: &[QueryEntry],
        dst_col: ColumnId,
    ) -> Result<Variable, QueryError> {
        let table_info = self.table_info(table);
        self.validate_keys(table, table_info, args)?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::Lookup {
            table,
            args: args.to_vec(),
            dst_col,
            dst_var: res,
        });
        self.qb.mark_used(args);
        self.qb.mark_defined(&res.into());
        Ok(res)
    }

    /// Insert the specified values into the given table.
    pub fn insert(&mut self, table: TableId, vals: &[QueryEntry]) -> Result<(), QueryError> {
        let table_info = self.table_info(table);
        self.validate_row(table, table_info, vals)?;
        self.qb.instrs.push(Instr::Insert {
            table,
            vals: vals.to_vec(),
        });
        self.qb.mark_used(vals);
        Ok(())
    }

    /// Insert the specified values into the given table if `l` and `r` are equal.
    pub fn insert_if_eq(
        &mut self,
        table: TableId,
        l: QueryEntry,
        r: QueryEntry,
        vals: &[QueryEntry],
    ) -> Result<(), QueryError> {
        let table_info = self.table_info(table);
        self.validate_row(table, table_info, vals)?;
        self.qb.instrs.push(Instr::InsertIfEq {
            table,
            l,
            r,
            vals: vals.to_vec(),
        });
        self.qb
            .mark_used(vals.iter().chain(once(&l)).chain(once(&r)));
        Ok(())
    }

    /// Remove the specified entry from the given table, if it is there.
    pub fn remove(&mut self, table: TableId, args: &[QueryEntry]) -> Result<(), QueryError> {
        let table_info = self.table_info(table);
        self.validate_keys(table, table_info, args)?;
        self.qb.instrs.push(Instr::Remove {
            table,
            args: args.to_vec(),
        });
        self.qb.mark_used(args);
        Ok(())
    }

    /// Apply the given external function to the specified arguments.
    pub fn call_external(
        &mut self,
        func: ExternalFunctionId,
        args: &[QueryEntry],
    ) -> Result<Variable, QueryError> {
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::External {
            func,
            args: args.to_vec(),
            dst: res,
        });
        self.qb.mark_used(args);
        self.qb.mark_defined(&res.into());
        Ok(res)
    }

    /// Look up the given key in the given table. If the lookup fails, then call the given external
    /// function with the given arguments. Bind the result to the returned variable. If the
    /// external function returns None (and the lookup fails) then the execution of the rule halts.
    pub fn lookup_with_fallback(
        &mut self,
        table: TableId,
        key: &[QueryEntry],
        dst_col: ColumnId,
        func: ExternalFunctionId,
        func_args: &[QueryEntry],
    ) -> Result<Variable, QueryError> {
        let table_info = self.table_info(table);
        self.validate_keys(table, table_info, key)?;
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::LookupWithFallback {
            table,
            table_key: key.to_vec(),
            func,
            func_args: func_args.to_vec(),
            dst_var: res,
            dst_col,
        });
        self.qb.mark_used(key);
        self.qb.mark_used(func_args);
        self.qb.mark_defined(&res.into());
        Ok(res)
    }

    pub fn call_external_with_fallback(
        &mut self,
        f1: ExternalFunctionId,
        args1: &[QueryEntry],
        f2: ExternalFunctionId,
        args2: &[QueryEntry],
    ) -> Result<Variable, QueryError> {
        let res = self.qb.new_var();
        self.qb.instrs.push(Instr::ExternalWithFallback {
            f1,
            args1: args1.to_vec(),
            f2,
            args2: args2.to_vec(),
            dst: res,
        });
        self.qb.mark_used(args1);
        self.qb.mark_used(args2);
        self.qb.mark_defined(&res.into());
        Ok(res)
    }

    /// Continue execution iff the two arguments are equal.
    pub fn assert_eq(&mut self, l: QueryEntry, r: QueryEntry) {
        self.qb.instrs.push(Instr::AssertEq(l, r));
        self.qb.mark_used(&[l, r]);
    }

    /// Continue execution iff the two arguments are not equal.
    pub fn assert_ne(&mut self, l: QueryEntry, r: QueryEntry) -> Result<(), QueryError> {
        self.qb.instrs.push(Instr::AssertNe(l, r));
        self.qb.mark_used(&[l, r]);
        Ok(())
    }

    /// Continue execution iff there is some `i` such that `l[i] != r[i]`.
    ///
    /// This is useful when doing egglog-style rebuilding.
    pub fn assert_any_ne(&mut self, l: &[QueryEntry], r: &[QueryEntry]) -> Result<(), QueryError> {
        if l.len() != r.len() {
            return Err(QueryError::MultiComparisonMismatch {
                l: l.len(),
                r: r.len(),
            });
        }

        let mut ops = Vec::with_capacity(l.len() + r.len());
        ops.extend_from_slice(l);
        ops.extend_from_slice(r);
        self.qb.instrs.push(Instr::AssertAnyNe {
            ops,
            divider: l.len(),
        });
        self.qb.mark_used(l);
        self.qb.mark_used(r);
        Ok(())
    }

    fn validate_row(
        &self,
        table: TableId,
        info: &TableInfo,
        vals: &[QueryEntry],
    ) -> Result<(), QueryError> {
        if vals.len() != info.spec.arity() {
            Err(QueryError::TableArityMismatch {
                table,
                expected: info.spec.arity(),
                got: vals.len(),
            })
        } else {
            Ok(())
        }
    }

    fn validate_keys(
        &self,
        table: TableId,
        info: &TableInfo,
        keys: &[QueryEntry],
    ) -> Result<(), QueryError> {
        if keys.len() != info.spec.n_keys {
            Err(QueryError::KeyArityMismatch {
                table,
                expected: info.spec.n_keys,
                got: keys.len(),
            })
        } else {
            Ok(())
        }
    }

    fn validate_vals<'b>(
        &self,
        table: TableId,
        info: &TableInfo,
        vals: impl Iterator<Item = &'b WriteVal>,
    ) -> Result<(), QueryError> {
        for (i, _) in vals.enumerate() {
            let col = i + info.spec.n_keys;
            if col >= info.spec.arity() {
                return Err(QueryError::TableArityMismatch {
                    table,
                    expected: info.spec.arity(),
                    got: col,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Atom {
    pub(crate) table: TableId,
    pub(crate) var_columns: VarColumnMap,
    /// These constraints are an initial take at processing "fast" constraints as well as a
    /// potential list of "slow" constraints.
    ///
    /// Fast constraints get re-computed when queries are executed. In particular, this makes it
    /// possible to cache plans and add new fast constraints to them without re-planning.
    pub(crate) constraints: ProcessedConstraints,
}

impl Atom {
    pub(crate) fn vars(&self) -> impl Iterator<Item = Variable> + '_ {
        self.var_columns.vars()
    }

    pub(crate) fn get_var(&self, col: ColumnId) -> Option<Variable> {
        self.var_columns.get_var(col)
    }

    pub(crate) fn get_col(&self, var: Variable) -> Option<ColumnId> {
        self.var_columns.get_col(var)
    }
}

#[derive(Clone, Default)]
pub(crate) struct VarColumnMap {
    var_to_column: DenseIdMap<Variable, ColumnId>,
    column_to_var: DenseIdMap<ColumnId, Variable>,
}

impl std::fmt::Debug for VarColumnMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut entries: Vec<_> = self.column_to_var.iter().collect();
        entries.sort_by_key(|(col, _)| col.index());

        f.write_str("VarColumnMap(")?;
        for (i, (col, var)) in entries.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{col:?} -> {var:?}")?;
        }
        f.write_str(")")
    }
}

impl VarColumnMap {
    pub(crate) fn insert(&mut self, var: Variable, col: ColumnId) -> Option<ColumnId> {
        let prev = self.var_to_column.insert(var, col);
        self.column_to_var.insert(col, var);
        prev
    }

    pub(crate) fn get_col(&self, var: Variable) -> Option<ColumnId> {
        self.var_to_column.get(var).copied()
    }

    pub(crate) fn get_var(&self, col: ColumnId) -> Option<Variable> {
        self.column_to_var.get(col).copied()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (ColumnId, Variable)> + '_ {
        self.column_to_var.iter().map(|(col, var)| (col, *var))
    }

    pub(crate) fn vars(&self) -> impl Iterator<Item = Variable> + '_ {
        self.iter().map(|(_, var)| var)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.var_to_column.len() == 0
    }
}

/// A functional dependency inferencer.
///
/// A functional dependency (x, y, ...) -> (u, v, ...) means that if we know
/// the values of x, y, ..., then we can determine u, v, ...
///
/// This data structure can compute the closure of a set of variables under
/// a set of functional dependencies.
#[derive(Clone, Default)]
pub(crate) struct FunDeps {
    /// List of functional dependencies (antecedent -> consequent)
    dependencies: Vec<(Vec<Variable>, Vec<Variable>)>,
}

impl FunDeps {
    /// Add a functional dependency: antecedent -> consequent.
    pub fn add_dependency(&mut self, antecedent: Vec<Variable>, consequent: Vec<Variable>) {
        // Don't add trivial dependencies.
        if !antecedent.is_empty() {
            self.dependencies.push((antecedent, consequent));
        }
    }

    /// Returns all variables that can be determined from the input variables
    /// using the functional dependencies.
    pub fn closure(
        &self,
        variables: impl IntoIterator<Item = Variable>,
    ) -> DenseIdMap<Variable, ()> {
        let mut result: DenseIdMap<Variable, ()> =
            DenseIdMap::from_iter(variables.into_iter().map(|v| (v, ())));
        let mut changed = true;

        while changed {
            changed = false;
            for (antecedent, consequent) in &self.dependencies {
                // If all variables in the antecedent are in the result,
                // add all variables in the consequent.
                if antecedent.iter().all(|v| result.contains_key(*v)) {
                    for v in consequent {
                        if !result.contains_key(*v) {
                            result.insert(*v, ());
                            changed = true;
                        }
                    }
                }
            }
        }

        result
    }
}

impl std::fmt::Debug for FunDeps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;

        let mut deps = String::new();

        for (i, (ant, cons)) in self.dependencies.iter().enumerate() {
            if i > 0 {
                deps.push_str("; ");
            }

            deps.push('{');
            for (j, v) in ant.iter().enumerate() {
                if j > 0 {
                    deps.push_str(", ");
                }
                write!(&mut deps, "{v:?}")?;
            }
            deps.push('}');

            deps.push_str(" -> ");

            deps.push('{');
            for (j, v) in cons.iter().enumerate() {
                if j > 0 {
                    deps.push_str(", ");
                }
                write!(&mut deps, "{v:?}")?;
            }
            deps.push('}');
        }

        write!(f, "FunDeps {{ {deps} }}")
    }
}

pub(crate) struct Query {
    pub(crate) var_info: DenseIdMap<Variable, VarInfo>,
    pub(crate) atoms: DenseIdMap<AtomId, Atom>,
    pub(crate) action: ActionId,
    pub(crate) plan_strategy: PlanStrategy,
    pub(crate) fun_deps: FunDeps,
    /// If `true`, skip tree-decomposition during query planning and
    /// always use the single-bag fast path in
    /// [`crate::free_join::plan::tree_decompose_and_plan`]. Set via
    /// [`QueryBuilder::set_no_decomp`].
    pub(crate) no_decomp: bool,
}
