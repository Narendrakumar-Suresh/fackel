import jax
from safetensors.flax import save_file, load_file


def save_checkpoint(tree, filepath):
    """
    Flattens a JAX PyTree and saves it using the safetensors format.
    """
    flat_tree, _ = jax.tree_util.tree_flatten_with_path(tree)

    flat_dict = {}
    for path, array in flat_tree:
        # Convert JAX dictionary path objects into dot-separated string keys
        key = ".".join([str(p.key) if hasattr(p, "key") else str(p.idx) for p in path])
        flat_dict[key] = array

    save_file(flat_dict, filepath)


def load_checkpoint(filepath, template_tree):
    """
    Loads a safetensors file and reconstructs the nested JAX PyTree.
    Requires a template_tree (e.g., from your init_fn) to know the exact structure.
    """
    flat_dict = load_file(filepath)
    flat_template, treedef = jax.tree_util.tree_flatten_with_path(template_tree)

    restored_flat = []
    for path, _ in flat_template:
        key = ".".join([str(p.key) if hasattr(p, "key") else str(p.idx) for p in path])
        if key not in flat_dict:
            raise KeyError(f"Key '{key}' missing from checkpoint.")
        restored_flat.append(flat_dict[key])

    return jax.tree_util.tree_unflatten(treedef, restored_flat)
