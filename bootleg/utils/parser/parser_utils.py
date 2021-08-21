"""Parses a Booleg input config into a DottedDict of config values (with
defaults filled in) for running a model."""

import argparse
import fileinput
import os

import ujson

import bootleg.utils.classes.comment_json as comment_json
from bootleg.utils.classes.dotted_dict import DottedDict, createBoolDottedDict
from bootleg.utils.parser.bootleg_args import config_args
from bootleg.utils.parser.emm_parse_args import (
    parse_args as emm_parse_args,
    parse_args_to_config as emm_parse_args_to_config,
)
from bootleg.utils.utils import load_yaml_file


def OrNone(default):
    def func(x):
        # Convert "none" to proper None object
        if x.lower() == "none":
            return None
        # If default is None (and x is not None), return x without conversion as str
        elif default is None:
            return str(x)
        # Treat bools separately as bool("False") is true
        elif isinstance(default, bool):
            if x.lower() == "false":
                return False
            return True
        # Otherwise, default has non-None type; convert x to that type
        else:
            return type(default)(x)

    return func


def is_number(s):
    """Returns True is string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_json(value):
    """Returns True if json."""
    # ujson is weird in that a string of a number is a dictionary; we don't want this
    if is_number(value):
        return False
    try:
        ujson.loads(value)
    except ValueError:
        return False
    return True


def recursive_keys(dictionary):
    """Recursively yields all keys of dict."""
    for key, value in dictionary.items():
        if type(value) is dict:
            yield key
            yield from recursive_keys(value)
        else:
            yield key


def merge_configs(config_l, config_r, new_config=None):
    """Merges two dotted dict configs."""
    if new_config is None:
        new_config = {}
    for k in config_l:
        # If unique to config_l or the same in both configs, add
        if k not in config_r or config_l[k] == config_r[k]:
            new_config[k] = config_l[k]
        # If not unique and different, then they must be dictionaries (that we can recursively merge)
        else:
            assert type(config_l[k]) in [dict, DottedDict] and type(config_r[k]) in [
                dict,
                DottedDict,
            ], f"You have two conflicting values for key {k}: {config_l[k]} vs {config_r[k]}"
            new_config[k] = merge_configs(config_l[k], config_r[k])
    for k in config_r:
        # If unique to config_r or the same in both configs, add
        if k not in config_l or config_l[k] == config_r[k]:
            new_config[k] = config_r[k]
    return new_config


def add_nested_flags_from_config(parser, config_dict, parser_hierarchy, prefix):
    """
    Add the flags from config file, keeping the hierarchy the same. When a lower level is needed,
     parser.add_argument_group is called. Note, we append the parent key to the --param option (via prefix parameter).
    Args:
        parser: arg parser to add options to
        config_dict: raw config dictionary
        parser_hierarchy: Dict to add parser hierarhcy to
        prefix: prefix to add to arg parser

    Returns:

    """
    for param in config_dict:
        if isinstance(config_dict[param], dict):
            parser_hierarchy[param] = {}
            temp = parser.add_argument_group(f"Bootleg specific {param.split('_')[0]}")
            add_nested_flags_from_config(
                temp, config_dict[param], parser_hierarchy[param], f"{prefix}{param}."
            )
        else:
            default, description = config_dict[param]
            try:
                if isinstance(default, str) and is_json(default):
                    parser.add_argument(
                        f"--{prefix}{param}",
                        type=ujson.loads,
                        default=default,
                        help=description,
                    )
                elif isinstance(default, list):
                    if len(default) > 0:
                        # pass a list as argument
                        parser.add_argument(
                            f"--{prefix}{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description,
                        )
                    else:
                        parser.add_argument(
                            f"--{prefix}{param}",
                            action="append",
                            default=default,
                            help=description,
                        )
                    parser_hierarchy["_global"] = parser
                else:
                    # pass
                    parser.add_argument(
                        f"--{prefix}{param}",
                        type=OrNone(default),
                        default=default,
                        help=description,
                    )
                    parser_hierarchy["_global"] = parser
            except argparse.ArgumentError:
                print(
                    f"Could not add flag for param {param} because it was already present."
                )
    return


def flatten_nested_args_for_parser(args, new_args, groups, prefix):
    """This will flatten all parameters to be passed as a single list to arg
    parse."""
    for key in args:
        if isinstance(args[key], dict):
            if key in groups:
                new_args = flatten_nested_args_for_parser(
                    args[key], new_args, groups, f"{prefix}{key}."
                )
            else:
                new_args.append(f"--{prefix}{key}")
                new_args.append(f"{ujson.dumps(vars(args)[key])}")
        elif isinstance(args[key], list):
            for v in args[key]:
                new_args.append(f"--{prefix}{key}")
                if isinstance(v, dict):
                    new_args.append(f"{ujson.dumps(v)}")
                else:
                    new_args.append(f"{v}")
        else:
            new_args.append(f"--{prefix}{key}")
            new_args.append(f"{args[key]}")
    return new_args


def reconstructed_nested_args(args, names, parser_hierarchy, prefix):
    """After getting the args back, we need to reconstruct the arguments and
    pass them to the necessary subparsers."""
    for key, sub_parser in parser_hierarchy.items():
        if isinstance(sub_parser, dict):
            names[key] = {}
            reconstructed_nested_args(args, names[key], sub_parser, f"{prefix}{key}.")
        else:
            sub_options = [action.dest for action in sub_parser._group_actions]
            sub_names = {
                name: value
                for (name, value) in args._get_kwargs()
                if name in sub_options
            }
            temp = argparse.Namespace(**sub_names)
            # remove the prefix from the key
            for k, v in temp.__dict__.items():
                names[k.replace(f"{prefix}", "")] = v
    return


def load_commented_json_file(file):
    """Load commented json file."""
    json_out = ""
    for line in fileinput.input(file):  # Read it all in
        json_out += line
    almost_json = comment_json.remove_comments(json_out)  # Remove comments
    proper_json = comment_json.remove_trailing_commas(
        almost_json
    )  # Remove trailing commas
    validated = ujson.loads(proper_json)  # We now have parseable JSON!
    return validated


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
        ], f"We only accept json or yaml ending for configs"
        if os.path.splitext(config)[1] == ".json":
            params = load_commented_json_file(config)
        else:
            params = load_yaml_file(config)
    else:
        assert (
            type(config) is dict
        ), f"We only support loading configs that are paths to json/yaml files or preloaded configs."
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
        assert not unknown[idx].startswith(("-", "--")) or (unknown[idx-1] == "emmental.device" and unknown[idx] == "-1")
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
