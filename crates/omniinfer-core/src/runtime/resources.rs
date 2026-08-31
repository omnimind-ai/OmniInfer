use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum MemoryDomain {
    Host,
    Cuda(String),
    Vulkan(String),
    Unified(String),
}

impl MemoryDomain {
    pub fn key(&self) -> String {
        match self {
            Self::Host => "host".to_string(),
            Self::Cuda(id) => format!("cuda:{id}"),
            Self::Vulkan(id) => format!("vulkan:{id}"),
            Self::Unified(id) => format!("unified:{id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetComponent {
    pub name: String,
    pub domain: MemoryDomain,
    pub bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    domains: BTreeMap<MemoryDomain, u64>,
    components: Vec<BudgetComponent>,
}

impl ResourceBudget {
    pub fn from_domains(domains: BTreeMap<MemoryDomain, u64>) -> Result<Self, ResourceLedgerError> {
        if domains.is_empty() || domains.values().any(|bytes| *bytes == 0) {
            return Err(ResourceLedgerError::InvalidBudget);
        }
        Ok(Self {
            domains,
            components: Vec::new(),
        })
    }

    pub fn from_components(components: Vec<BudgetComponent>) -> Result<Self, ResourceLedgerError> {
        if components.is_empty() || components.iter().any(|component| component.bytes == 0) {
            return Err(ResourceLedgerError::InvalidBudget);
        }
        let mut domains = BTreeMap::new();
        for component in &components {
            let total = domains.entry(component.domain.clone()).or_insert(0_u64);
            *total = total
                .checked_add(component.bytes)
                .ok_or(ResourceLedgerError::ArithmeticOverflow)?;
        }
        Ok(Self {
            domains,
            components,
        })
    }

    pub fn domains(&self) -> &BTreeMap<MemoryDomain, u64> {
        &self.domains
    }

    pub fn components(&self) -> &[BudgetComponent] {
        &self.components
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceCapacity {
    pub snapshot_id: u64,
    pub domains: BTreeMap<MemoryDomain, u64>,
}

impl ResourceCapacity {
    pub fn new(
        snapshot_id: u64,
        domains: BTreeMap<MemoryDomain, u64>,
    ) -> Result<Self, ResourceLedgerError> {
        if snapshot_id == 0 || domains.is_empty() {
            return Err(ResourceLedgerError::InvalidCapacity);
        }
        Ok(Self {
            snapshot_id,
            domains,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReservationId(u64);

impl ReservationId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AllocationId(u64);

impl AllocationId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResourceRecord {
    request_id: String,
    budget: ResourceBudget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLedgerSnapshot {
    pub capacity_snapshot_id: u64,
    pub capacities: BTreeMap<MemoryDomain, u64>,
    pub reserved: BTreeMap<MemoryDomain, u64>,
    pub committed: BTreeMap<MemoryDomain, u64>,
}

impl ResourceLedgerSnapshot {
    pub fn available(&self) -> Result<BTreeMap<MemoryDomain, u64>, ResourceLedgerError> {
        let mut available = self.capacities.clone();
        subtract_domains(&mut available, &self.reserved)?;
        subtract_domains(&mut available, &self.committed)?;
        Ok(available)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResourceLedgerError {
    #[error("resource budget must contain non-zero domain requirements")]
    InvalidBudget,
    #[error("resource capacity must have a non-zero snapshot and at least one domain")]
    InvalidCapacity,
    #[error("capacity snapshot {received} is not newer than {current}")]
    StaleCapacitySnapshot { current: u64, received: u64 },
    #[error("capacity for {domain} is below resources already reserved or committed")]
    CapacityBelowUsage { domain: String },
    #[error(
        "insufficient {domain} memory: requested {requested} bytes, available {available} bytes"
    )]
    InsufficientCapacity {
        domain: String,
        requested: u64,
        available: u64,
    },
    #[error("request already owns resources: {0}")]
    DuplicateRequest(String),
    #[error("unknown reservation: {0}")]
    UnknownReservation(u64),
    #[error("resource ledger arithmetic overflow")]
    ArithmeticOverflow,
    #[error("resource ledger invariant failed for {0}")]
    InvariantViolation(String),
}

#[derive(Debug)]
pub struct ResourceLedger {
    capacity: ResourceCapacity,
    reserved: BTreeMap<ReservationId, ResourceRecord>,
    committed: BTreeMap<AllocationId, ResourceRecord>,
    next_id: u64,
}

impl ResourceLedger {
    pub fn new(capacity: ResourceCapacity) -> Self {
        Self {
            capacity,
            reserved: BTreeMap::new(),
            committed: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn update_capacity(
        &mut self,
        capacity: ResourceCapacity,
    ) -> Result<(), ResourceLedgerError> {
        if capacity.snapshot_id <= self.capacity.snapshot_id {
            return Err(ResourceLedgerError::StaleCapacitySnapshot {
                current: self.capacity.snapshot_id,
                received: capacity.snapshot_id,
            });
        }
        let usage = self.total_usage()?;
        for (domain, used) in usage {
            if capacity.domains.get(&domain).copied().unwrap_or(0) < used {
                return Err(ResourceLedgerError::CapacityBelowUsage {
                    domain: domain.key(),
                });
            }
        }
        self.capacity = capacity;
        Ok(())
    }

    pub fn reserve(
        &mut self,
        request_id: impl Into<String>,
        budget: ResourceBudget,
    ) -> Result<ReservationId, ResourceLedgerError> {
        let request_id = request_id.into();
        if self.request_ids().contains(request_id.as_str()) {
            return Err(ResourceLedgerError::DuplicateRequest(request_id));
        }
        let available = self.snapshot().available()?;
        for (domain, requested) in budget.domains() {
            let remaining = available.get(domain).copied().unwrap_or(0);
            if *requested > remaining {
                return Err(ResourceLedgerError::InsufficientCapacity {
                    domain: domain.key(),
                    requested: *requested,
                    available: remaining,
                });
            }
        }
        let reservation_id = ReservationId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ResourceLedgerError::ArithmeticOverflow)?;
        self.reserved
            .insert(reservation_id, ResourceRecord { request_id, budget });
        Ok(reservation_id)
    }

    pub fn commit(
        &mut self,
        reservation_id: ReservationId,
    ) -> Result<AllocationId, ResourceLedgerError> {
        let record = self.reserved.remove(&reservation_id).ok_or(
            ResourceLedgerError::UnknownReservation(reservation_id.get()),
        )?;
        let allocation_id = AllocationId(reservation_id.get());
        self.committed.insert(allocation_id, record);
        Ok(allocation_id)
    }

    /// Atomically replaces a provisional reservation with a reconciled budget.
    ///
    /// Capacity is checked while excluding the reservation being replaced, so a
    /// runtime can safely shrink, grow, or move its budget across domains after
    /// startup without briefly releasing its concurrency guard.
    pub fn reconcile_reservation(
        &mut self,
        reservation_id: ReservationId,
        budget: ResourceBudget,
    ) -> Result<(), ResourceLedgerError> {
        let current =
            self.reserved
                .get(&reservation_id)
                .ok_or(ResourceLedgerError::UnknownReservation(
                    reservation_id.get(),
                ))?;
        let mut other_usage = self.total_usage()?;
        subtract_domains(&mut other_usage, current.budget.domains())?;
        let mut available = self.capacity.domains.clone();
        subtract_domains(&mut available, &other_usage)?;
        for (domain, requested) in budget.domains() {
            let remaining = available.get(domain).copied().unwrap_or(0);
            if *requested > remaining {
                return Err(ResourceLedgerError::InsufficientCapacity {
                    domain: domain.key(),
                    requested: *requested,
                    available: remaining,
                });
            }
        }
        self.reserved
            .get_mut(&reservation_id)
            .expect("validated reservation must still exist")
            .budget = budget;
        Ok(())
    }

    pub fn rollback(&mut self, reservation_id: ReservationId) -> bool {
        self.reserved.remove(&reservation_id).is_some()
    }

    pub fn release(&mut self, allocation_id: AllocationId) -> bool {
        self.committed.remove(&allocation_id).is_some()
    }

    pub fn snapshot(&self) -> ResourceLedgerSnapshot {
        ResourceLedgerSnapshot {
            capacity_snapshot_id: self.capacity.snapshot_id,
            capacities: self.capacity.domains.clone(),
            reserved: sum_records(self.reserved.values())
                .expect("validated budgets cannot overflow ledger totals"),
            committed: sum_records(self.committed.values())
                .expect("validated budgets cannot overflow ledger totals"),
        }
    }

    fn total_usage(&self) -> Result<BTreeMap<MemoryDomain, u64>, ResourceLedgerError> {
        let mut usage = sum_records(self.reserved.values())?;
        add_domains(&mut usage, &sum_records(self.committed.values())?)?;
        Ok(usage)
    }

    fn request_ids(&self) -> BTreeSet<&str> {
        self.reserved
            .values()
            .chain(self.committed.values())
            .map(|record| record.request_id.as_str())
            .collect()
    }
}

fn sum_records<'a>(
    records: impl Iterator<Item = &'a ResourceRecord>,
) -> Result<BTreeMap<MemoryDomain, u64>, ResourceLedgerError> {
    let mut totals = BTreeMap::new();
    for record in records {
        for (domain, bytes) in record.budget.domains() {
            let total = totals.entry(domain.clone()).or_insert(0_u64);
            *total = total
                .checked_add(*bytes)
                .ok_or(ResourceLedgerError::ArithmeticOverflow)?;
        }
    }
    Ok(totals)
}

fn add_domains(
    target: &mut BTreeMap<MemoryDomain, u64>,
    values: &BTreeMap<MemoryDomain, u64>,
) -> Result<(), ResourceLedgerError> {
    for (domain, bytes) in values {
        let total = target.entry(domain.clone()).or_insert(0);
        *total = total
            .checked_add(*bytes)
            .ok_or(ResourceLedgerError::ArithmeticOverflow)?;
    }
    Ok(())
}

fn subtract_domains(
    target: &mut BTreeMap<MemoryDomain, u64>,
    values: &BTreeMap<MemoryDomain, u64>,
) -> Result<(), ResourceLedgerError> {
    for (domain, bytes) in values {
        let available = target.get(domain).copied().unwrap_or(0);
        let remaining = available
            .checked_sub(*bytes)
            .ok_or_else(|| ResourceLedgerError::InvariantViolation(domain.key()))?;
        target.insert(domain.clone(), remaining);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn domains(entries: &[(MemoryDomain, u64)]) -> BTreeMap<MemoryDomain, u64> {
        entries.iter().cloned().collect()
    }

    fn ledger(entries: &[(MemoryDomain, u64)]) -> ResourceLedger {
        ResourceLedger::new(ResourceCapacity::new(1, domains(entries)).unwrap())
    }

    fn budget(entries: &[(MemoryDomain, u64)]) -> ResourceBudget {
        ResourceBudget::from_domains(domains(entries)).unwrap()
    }

    #[test]
    fn reservation_prevents_oversubscription_and_rollback_restores_capacity() {
        let mut ledger = ledger(&[(MemoryDomain::Host, 10 * GIB)]);
        let first = ledger
            .reserve("load-a", budget(&[(MemoryDomain::Host, 7 * GIB)]))
            .unwrap();

        assert!(matches!(
            ledger.reserve("load-b", budget(&[(MemoryDomain::Host, 7 * GIB)])),
            Err(ResourceLedgerError::InsufficientCapacity { .. })
        ));
        assert!(ledger.rollback(first));
        assert!(!ledger.rollback(first));
        assert!(
            ledger
                .reserve("load-b", budget(&[(MemoryDomain::Host, 7 * GIB)]))
                .is_ok()
        );
    }

    #[test]
    fn commit_moves_usage_and_release_is_idempotent() {
        let mut ledger = ledger(&[(MemoryDomain::Host, 10 * GIB)]);
        let reservation = ledger
            .reserve("load-a", budget(&[(MemoryDomain::Host, 7 * GIB)]))
            .unwrap();
        let allocation = ledger.commit(reservation).unwrap();
        let snapshot = ledger.snapshot();

        assert!(snapshot.reserved.is_empty());
        assert_eq!(snapshot.committed[&MemoryDomain::Host], 7 * GIB);
        assert_eq!(snapshot.available().unwrap()[&MemoryDomain::Host], 3 * GIB);
        assert!(ledger.release(allocation));
        assert!(!ledger.release(allocation));
        assert_eq!(
            ledger.snapshot().available().unwrap()[&MemoryDomain::Host],
            10 * GIB
        );
    }

    #[test]
    fn multi_domain_reservation_is_all_or_nothing() {
        let cuda = MemoryDomain::Cuda("0".to_string());
        let mut ledger = ledger(&[(MemoryDomain::Host, 8 * GIB), (cuda.clone(), 4 * GIB)]);
        let result = ledger.reserve(
            "load-a",
            budget(&[(MemoryDomain::Host, 2 * GIB), (cuda.clone(), 5 * GIB)]),
        );

        assert!(matches!(
            result,
            Err(ResourceLedgerError::InsufficientCapacity { domain, .. }) if domain == "cuda:0"
        ));
        assert!(ledger.snapshot().reserved.is_empty());
        assert_eq!(
            ledger.snapshot().available().unwrap()[&MemoryDomain::Host],
            8 * GIB
        );
    }

    #[test]
    fn capacity_updates_reject_stale_snapshots_and_allocated_underflow() {
        let mut ledger = ledger(&[(MemoryDomain::Host, 10 * GIB)]);
        let reservation = ledger
            .reserve("load-a", budget(&[(MemoryDomain::Host, 7 * GIB)]))
            .unwrap();
        ledger.commit(reservation).unwrap();

        assert!(matches!(
            ledger.update_capacity(
                ResourceCapacity::new(1, domains(&[(MemoryDomain::Host, 12 * GIB)])).unwrap()
            ),
            Err(ResourceLedgerError::StaleCapacitySnapshot { .. })
        ));
        assert!(matches!(
            ledger.update_capacity(
                ResourceCapacity::new(2, domains(&[(MemoryDomain::Host, 6 * GIB)])).unwrap()
            ),
            Err(ResourceLedgerError::CapacityBelowUsage { .. })
        ));
        assert_eq!(ledger.snapshot().capacity_snapshot_id, 1);
    }

    #[test]
    fn component_totals_detect_overflow() {
        let result = ResourceBudget::from_components(vec![
            BudgetComponent {
                name: "weights".to_string(),
                domain: MemoryDomain::Host,
                bytes: u64::MAX,
            },
            BudgetComponent {
                name: "kv_cache".to_string(),
                domain: MemoryDomain::Host,
                bytes: 1,
            },
        ]);

        assert_eq!(result, Err(ResourceLedgerError::ArithmeticOverflow));
    }

    #[test]
    fn reservation_reconciliation_is_atomic_across_domains() {
        let cuda = MemoryDomain::Cuda("0".to_string());
        let mut ledger = ledger(&[(MemoryDomain::Host, 10 * GIB), (cuda.clone(), 8 * GIB)]);
        let owner = ledger
            .reserve(
                "owner",
                budget(&[(MemoryDomain::Host, 8 * GIB), (cuda.clone(), 8 * GIB)]),
            )
            .unwrap();

        ledger
            .reconcile_reservation(
                owner,
                budget(&[(MemoryDomain::Host, 4 * GIB), (cuda.clone(), 6 * GIB)]),
            )
            .unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.reserved[&MemoryDomain::Host], 4 * GIB);
        assert_eq!(snapshot.reserved[&cuda], 6 * GIB);

        let failed = ledger.reconcile_reservation(
            owner,
            budget(&[(MemoryDomain::Host, 11 * GIB), (cuda.clone(), GIB)]),
        );
        assert!(matches!(
            failed,
            Err(ResourceLedgerError::InsufficientCapacity { domain, .. }) if domain == "host"
        ));
        let unchanged = ledger.snapshot();
        assert_eq!(unchanged.reserved[&MemoryDomain::Host], 4 * GIB);
        assert_eq!(unchanged.reserved[&cuda], 6 * GIB);
    }

    #[test]
    fn reconciliation_respects_other_reservations() {
        let mut ledger = ledger(&[(MemoryDomain::Host, 10 * GIB)]);
        let first = ledger
            .reserve("first", budget(&[(MemoryDomain::Host, 4 * GIB)]))
            .unwrap();
        ledger
            .reserve("second", budget(&[(MemoryDomain::Host, 5 * GIB)]))
            .unwrap();

        assert!(matches!(
            ledger.reconcile_reservation(first, budget(&[(MemoryDomain::Host, 6 * GIB)])),
            Err(ResourceLedgerError::InsufficientCapacity { available, .. }) if available == 5 * GIB
        ));
    }
}
