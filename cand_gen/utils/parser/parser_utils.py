"""Parses a Booleg input config into a DottedDict of config values (with
defaults filled in) for running a model."""

import argparse
import os

from bootleg.utils.classes.dotted_dict import createBoolDottedDict
from bootleg.utils.parser.emm_parse_args import (
    parse_args as emm_parse_args,
    parse_args_to_config as emm_parse_args_to_config,
)
from bootleg.utils.parser.parser_utils import (
    add_nested_flags_from_config,
    flatten_nested_args_for_parser,
    load_commented_json_file,
    merge_configs,
    reconstructed_nested_args,
    recursive_keys,
)
from bootleg.utils.utils import load_yaml_file
from cand_gen.utils.parser.candgen_args import config_args


def get_boot_config(config, parser_hierarchy=None, parser=None, unknown=None):
    """
    Returns a parsed Bootleg config from config. Config can be a path to a config file or an already loaded dictionary.
    The high level work flow
       1. Reads Bootleg default config (config_args) and addes params to a arg parser,
          flattening all hierarchical values into "." values
               E.g., data_config -> word_embeddings -> layers becomes --data_config.word_embedding.layers
       2. Flattens the given config values into the "." format
       3. Adds any unknown values from the first arg parser that parses the config script.
          Allows the user to add --data_config.word_embedding.layers to command line that overwrite values in file
       4. Parses the flattened args w.r.t the arg parser
       5. Reconstruct the args back into their hierarchical form
    Args:
        config: model specific config
        parser_hierarchy: Dict of hierarchy of config (or None)
        parser: arg parser (or None)
        unknown: unknown arg values passed from command line to be added to config and overwrite values in file

    Returns: parsed config

    """
    if unknown is None:
        unknown = []
    if parser_hierarchy is None:
        parser_hierarchy = {}
    if parser is None:
        parser = argparse.ArgumentParser()

    add_nested_flags_from_config(parser, config_args, parser_hierarchy, prefix="")
    if type(config) is str:
        assert os.path.splitext(config)[1] in [
            ".json",
            ".yaml",
        ], "We only accept json or yaml ending for configs"
        if os.path.splitext(config)[1] == ".json":
            params = load_commented_json_file(config)
        else:
            params = load_yaml_file(config)
    else:
        assert (
            type(config) is dict
        ), "We only support loading configs that are paths to json/yaml files or preloaded configs."
        params = config
    all_keys = list(recursive_keys(parser_hierarchy))
    new_params = flatten_nested_args_for_parser(params, [], groups=all_keys, prefix="")
    # update with new args
    # unknown must have ["--arg1", "value1", "--arg2", "value2"] as we don't have any action_true args
    assert len(unknown) % 2 == 0
    assert all(
        unknown[idx].startswith(("-", "--")) for idx in range(0, len(unknown), 2)
    )
    for idx in range(1, len(unknown), 2):
        # allow passing -1 for emmental.device argument
        assert not unknown[idx].startswith(("-", "--")) or (
            unknown[idx - 1] == "emmental.device" and unknown[idx] == "-1"
        )
    for idx in range(0, len(unknown), 2):
        arg = unknown[idx]
        # If override one you already have in json
        if arg in new_params:
            idx2 = new_params.index(arg)
            new_params[idx2 : idx2 + 2] = unknown[idx : idx + 2]
        # If override one that is in bootleg_args.py by not in json
        else:
            new_params.extend(unknown[idx : idx + 2])
    args = parser.parse_args(new_params)
    top_names = {}
    reconstructed_nested_args(args, top_names, parser_hierarchy, prefix="")
    # final_args = argparse.Namespace(**top_names)
    final_args = createBoolDottedDict(top_names)
    # turn_to_dotdicts(final_args)
    return final_args


def parse_boot_and_emm_args(config_script, unknown=None):
    """
    Merges the Emmental config with the Bootleg config.
    As we have an emmental: ... level in our config for emmental commands,
    we need to parse those with the Emmental parser and then merge the Bootleg only config values
    with the Emmental ones.
    Args:
        config_script: config script for Bootleg and Emmental args
        unknown: unknown arg values passed from command line to overwrite file values

    Returns: parsed merged Bootleg and Emmental config

    """
    if unknown is None:
        unknown = []
    config_parser = argparse.ArgumentParser(
        description="Bootleg Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Modified parse_args to have 'emmental.group' prefixes. This represents a hierarchy in our parser
    config_parser, parser_hierarchy = emm_parse_args(parser=config_parser)
    # Add Bootleg args and parse
    all_args = get_boot_config(config_script, parser_hierarchy, config_parser, unknown)
    # These have emmental -> config group -> arg structure for emmental.
    # Must remove that hierarchy to converte to internal Emmental hierarchy
    emm_args = {}
    for k, v in all_args["emmental"].items():
        emm_args[k] = v
    del all_args["emmental"]
    # create and add Emmental hierarchy
    config = emm_parse_args_to_config(createBoolDottedDict(emm_args))
    # Merge configs back (merge workds on dicts so must convert to dict first)
    config = createBoolDottedDict(merge_configs(all_args, config))
    return config
