//! What a mixed-precision allocation did, per component, and how faithful it is.
//!
//! A baseline nobody can inspect is not a baseline. `Q4_K_M` names an upstream
//! allocation, so a file labelled `Q4_K_M` carries an implicit claim: these are
//! the tensors llama.cpp would have promoted. compressr can only make that
//! claim for a model shaped the way upstream's rules assume — one transformer
//! stack — and this module states, in the conversion output, exactly which
//! numbers the rules were given and where each of them came from.

use super::config_facts::LayerCountSource;
use super::mix_components::UNLAYERED_COMPONENT;
use std::collections::BTreeMap;

/// One transformer stack's share of an allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentAllocation {
    /// The stack's tensor-name prefix, or `(unlayered)`
    pub name: String,
    /// The `n` every `use_more_bits(i, n)` call for this stack was given
    pub layer_count: usize,
    /// Where that `n` came from
    pub layer_count_source: String,
    /// The denominator the attention-V rule was given for this stack
    pub attn_v_denominator: usize,
    /// Tensors in this stack
    pub tensors: usize,
    /// Tensors a mix rule assigned a block format to
    pub quantized: usize,
    /// Tensors a rule moved OFF the mix's base format
    pub promoted: usize,
    /// Tensors no mix quantizes: norms, routers, biases, excluded names
    pub unquantized: usize,
    /// How many promoted tensors landed in each format, by format name
    pub promoted_formats: BTreeMap<String, usize>,
}

/// The inputs a mixed-precision allocation was computed from, and its result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixAllocation {
    /// The mix, as llama.cpp names it
    pub mix_name: String,
    /// The format every quantizable tensor takes unless a rule moves it
    pub base_format: String,
    /// The model-wide layer count, as stated or inferred
    pub model_layer_count: usize,
    /// Where that count was read
    pub layer_count_source: LayerCountSource,
    /// Routed experts the rules were told about
    pub n_expert: usize,
    /// One row per transformer stack, in checkpoint order
    pub components: Vec<ComponentAllocation>,
}

impl MixAllocation {
    /// The stacks that carry a layer index, which is what the rules divide by.
    pub fn layered_components(&self) -> impl Iterator<Item = &ComponentAllocation> + '_ {
        self.components
            .iter()
            .filter(|component| component.name != UNLAYERED_COMPONENT && component.layer_count > 0)
    }

    /// Whether this model has more than one transformer stack.
    ///
    /// llama.cpp's rules were written for a model where this is false. When it
    /// is true the allocation is an approximation, and says so.
    pub fn is_multi_component(&self) -> bool {
        self.layered_components().count() > 1
    }

    /// Promoted tensors over the whole model.
    pub fn total_promoted(&self) -> usize {
        self.components.iter().map(|c| c.promoted).sum()
    }

    /// The fidelity statement that belongs in the report.
    ///
    /// A single-stack model gets the upstream allocation and says so. A
    /// multi-stack model does not, and the difference is named rather than
    /// summarised.
    pub fn fidelity(&self) -> String {
        if !self.is_multi_component() {
            return format!(
                "Fidelity: this is llama.cpp's {} allocation. One transformer stack of {} \
                 layers, which is the shape `llama_tensor_get_type` assumes.",
                self.mix_name, self.model_layer_count
            );
        }
        let stacks: Vec<String> = self
            .layered_components()
            .map(|component| format!("{} ({} layers)", component.name, component.layer_count))
            .collect();
        format!(
            "Fidelity: this is an APPROXIMATION of llama.cpp's {mix}, not that allocation. \
             `llama_tensor_get_type` assumes one transformer stack of n layers indexed \
             0..n, and this model has {count}: {stacks}. llama.cpp has no allocation for \
             this architecture, so compressr made a choice, and it is compressr's, not \
             upstream's: `use_more_bits(i, n)` is applied ONCE PER STACK, with that stack's \
             own layer index as i and its own layer count as n. It therefore protects the \
             first eighth, the last eighth and every third middle layer OF EACH STACK. \
             Applying the rule once over a combined ordering was rejected: it would put \
             every layer of every stack after the first into the \"last eighth\" branch and \
             promote them all. The per-role assignments themselves are unchanged from \
             upstream. Anything above that llama.cpp does not specify is listed as \
             compressr's choice in the per-component table.",
            mix = self.mix_name,
            count = stacks.len(),
            stacks = stacks.join(", "),
        )
    }
}

impl std::fmt::Display for MixAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Mixed-precision allocation {} (base {}), {} expert(s)",
            self.mix_name, self.base_format, self.n_expert
        )?;
        writeln!(
            f,
            "  model layer count {}, read from {}",
            self.model_layer_count, self.layer_count_source
        )?;
        writeln!(
            f,
            "  {:<32} {:>6} {:>7} {:>9} {:>8}  promoted formats",
            "component", "layers", "tensors", "promoted", "unquant"
        )?;
        for component in &self.components {
            let formats: Vec<String> = component
                .promoted_formats
                .iter()
                .map(|(format, count)| format!("{format}x{count}"))
                .collect();
            writeln!(
                f,
                "  {:<32} {:>6} {:>7} {:>9} {:>8}  {}",
                component.name,
                component.layer_count,
                component.tensors,
                component.promoted,
                component.unquantized,
                if formats.is_empty() {
                    "-".to_string()
                } else {
                    formats.join(" ")
                }
            )?;
        }
        for component in &self.components {
            writeln!(
                f,
                "  n for {}: {} from {}; attention-V denominator {}",
                component.name,
                component.layer_count,
                component.layer_count_source,
                component.attn_v_denominator
            )?;
        }
        write!(f, "  {}", self.fidelity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component(name: &str, layers: usize, promoted: usize) -> ComponentAllocation {
        ComponentAllocation {
            name: name.into(),
            layer_count: layers,
            layer_count_source: "tensor names".into(),
            attn_v_denominator: layers,
            tensors: layers * 7,
            quantized: layers * 6,
            promoted,
            unquantized: layers,
            promoted_formats: BTreeMap::from([("Q6_K".to_string(), promoted)]),
        }
    }

    fn allocation(components: Vec<ComponentAllocation>) -> MixAllocation {
        MixAllocation {
            mix_name: "Q4_K_M".into(),
            base_format: "Q4_K".into(),
            model_layer_count: 28,
            layer_count_source: LayerCountSource::Config("lm_config.num_hidden_layers".into()),
            n_expert: 0,
            components,
        }
    }

    /// A single stack IS the shape upstream wrote the rules for, so the report
    /// claims the upstream allocation outright.
    #[test]
    fn a_single_stack_claims_the_upstream_allocation() {
        let report = allocation(vec![component("model.layers", 28, 8)]);
        assert!(!report.is_multi_component());
        assert!(report
            .fidelity()
            .starts_with("Fidelity: this is llama.cpp's"));
        assert!(!report.fidelity().contains("APPROXIMATION"));
    }

    /// Several stacks are a shape upstream's rules were not written for, and
    /// the report must say that in those words rather than implying a match.
    #[test]
    fn several_stacks_are_reported_as_an_approximation() {
        let report = allocation(vec![
            component("base_lm.layers", 28, 8),
            component("residual_lm.layers", 8, 4),
        ]);
        assert!(report.is_multi_component());

        let fidelity = report.fidelity();
        assert!(fidelity.contains("APPROXIMATION"), "{fidelity}");
        assert!(fidelity.contains("ONCE PER STACK"), "{fidelity}");
        assert!(
            fidelity.contains("base_lm.layers (28 layers)"),
            "{fidelity}"
        );
        assert!(
            fidelity.contains("residual_lm.layers (8 layers)"),
            "{fidelity}"
        );
        assert_eq!(report.total_promoted(), 12);
    }

    /// The unlayered tensors are a row in the table but not a stack: they
    /// divide by nothing, so they cannot make an allocation multi-component.
    #[test]
    fn the_unlayered_row_is_not_a_stack() {
        let mut unlayered = component("(unlayered)", 0, 1);
        unlayered.tensors = 3;
        let report = allocation(vec![unlayered, component("model.layers", 28, 8)]);
        assert!(!report.is_multi_component());
        assert_eq!(report.layered_components().count(), 1);
    }

    /// Every number a rule was given appears in the printed report, so an
    /// allocation can be checked by hand against `llama_tensor_get_type`.
    #[test]
    fn the_printed_report_names_every_input() {
        let report = allocation(vec![
            component("base_lm.layers", 28, 8),
            component("residual_lm.layers", 8, 4),
        ]);
        let text = report.to_string();

        assert!(text.contains("Q4_K_M"), "{text}");
        assert!(
            text.contains("config.json:lm_config.num_hidden_layers"),
            "{text}"
        );
        assert!(text.contains("n for base_lm.layers: 28"), "{text}");
        assert!(text.contains("n for residual_lm.layers: 8"), "{text}");
        assert!(text.contains("attention-V denominator"), "{text}");
        assert!(text.contains("Q6_Kx8"), "{text}");
    }
}
