//! Input type specification for diagrams.

/// Specifies how input values should be interpreted.
///
/// This determines whether the provided values represent disjoint regions
/// or complete set sizes including overlaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputType {
    /// Values represent disjoint regions that sum to the total set size.
    ///
    /// For example: `A=5, B=2, A&B=1` means set A has total size 6 (5+1)
    /// and set B has total size 3 (2+1).
    Disjoint,

    /// Values represent complete set sizes including overlaps.
    ///
    /// For example: `A=6, B=3, A&B=1` means set A has total size 6
    /// and set B has total size 3.
    #[default]
    Union,
}
