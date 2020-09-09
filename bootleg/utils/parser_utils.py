import argparse
import fileinput

import ujson

from bootleg.config import config_args
import bootleg.utils.classes.comment_json as comment_json
from bootleg.utils.classes.dotted_dict import DottedDict

"""
Adds a flag (and default value) to an ArgumentParser for each parameter in a config
"""
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
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_json(value):
    # ujson is weird in that a string of a number is a dictionary; we don't want this
    if is_number(value):
        return False
    try:
        ujson.loads(value)
    except ValueError as e:
        return False
    return True


def recursive_keys(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield key
            yield from recursive_keys(value)
        else:
            yield key


def createdBoolDottedDict(d_dict):
    if (type(d_dict) is DottedDict) or (type(d_dict) is dict):
        d_dict = DottedDict(d_dict)
    if type(d_dict) is str and is_json(d_dict):
        d_dict = DottedDict(ujson.loads(d_dict))
    if type(d_dict) is DottedDict:
        for k in d_dict:
            if d_dict[k] == "True":
                d_dict[k] = True
            elif d_dict[k] == "False":
                d_dict[k] = False
            elif (type(d_dict[k]) is DottedDict) or (type(d_dict[k]) is dict) or (type(d_dict[k]) is str and is_json(d_dict[k])):
                d_dict[k] = createdBoolDottedDict(d_dict[k])
            elif type(d_dict[k]) is list:
                for i in range(len(d_dict[k])):
                    d_dict[k][i] = createdBoolDottedDict(d_dict[k][i])
    return d_dict

# Add the flags from config file, keeping the hierarchy the same. When a lower level is needed, parser.add_argument_group is called. Note, we append
# the parent key to the --param option (via prefix parameter).
def add_nested_flags_from_config(parser, config_dict, sub_parsers, prefix):
    for param in config_dict:
        if isinstance(config_dict[param], dict):
            sub_parsers[param] = {}
            temp = parser.add_argument_group(param)
            add_nested_flags_from_config(temp, config_dict[param], sub_parsers[param], f"{prefix}{param}.")
        else:
            default, description = config_dict[param]
            try:
                if isinstance(default, str) and is_json(default):
                    parser.add_argument(f"--{prefix}{param}", type=ujson.loads, default=default, help=description)
                elif isinstance(default, list):
                    if len(default) > 0:
                        # pass a list as argument
                        parser.add_argument(
                                f"--{prefix}{param}",
                                action="append",
                                type=type(default[0]),
                                default=default,
                                help=description
                        )
                    else:
                        parser.add_argument(f"--{prefix}{param}", action="append", default=default, help=description)
                    sub_parsers["_global"] = parser
                else:
                    # pass
                    parser.add_argument(f"--{prefix}{param}", type=OrNone(default), default=default, help=description)
                    sub_parsers["_global"] = parser
            except argparse.ArgumentError:
                print(
                    f"Could not add flag for param {param} because it was already present."
                )
    return

# This will flatten all parameters to be passed as a single list to arg parse.
def flatten_nested_args_for_parser(args, new_args, groups, prefix):
    for key in args:
        if isinstance(args[key], dict):
            if key in groups:
                new_args = flatten_nested_args_for_parser(args[key], new_args, groups, f"{prefix}{key}.")
            else:
                new_args.append(f"--{prefix}{key}")
                new_args.append(f"{ujson.dumps(args[key])}")
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

# After getting the args back, we need to reconstruct the arguments and pass them to the necessary subparsers
def reconstructed_nested_args(args, names, sub_parsers, prefix):
    for key, sub_parser in sub_parsers.items():
        if isinstance(sub_parser, dict):
            names[key] = {}
            reconstructed_nested_args(args, names[key], sub_parser, f"{prefix}{key}.")
        else:
            sub_options = [action.dest for action in sub_parser._group_actions]
            sub_names = {name: value for (name, value) in args._get_kwargs() if name in sub_options}
            temp = argparse.Namespace(**sub_names)
            # remove the prefix from the key
            for k, v in temp.__dict__.items():
                names[k.replace(f"{prefix}", "")] = v
    return


# Load json file, ignoring commented lines
def load_commented_json_file(file):
    json_out = ""
    for line in fileinput.input(file): # Read it all in
        json_out += line
    almost_json = comment_json.remove_comments(json_out) # Remove comments
    proper_json = comment_json.remove_trailing_commas(almost_json) # Remove trailing commas
    validated = ujson.loads(proper_json) # We now have parseable JSON!
    return validated

def get_full_config(config_file, unknown=[]):
    parser = argparse.ArgumentParser()
    # start argparser with default args
    sub_parsers = {}
    add_nested_flags_from_config(parser, config_args, sub_parsers, prefix="")
    params = load_commented_json_file(config_file)
    all_keys = list(recursive_keys(sub_parsers))
    new_params = flatten_nested_args_for_parser(params, [], groups=all_keys, prefix="")
    # update with new args
    # unknown must have ["--arg1", "value1", "--arg2", "value2"] as we don't have any action_true args
    assert len(unknown) % 2 == 0
    assert all(unknown[idx].startswith(("-", "--")) for idx in range(0, len(unknown), 2))
    assert all(not unknown[idx].startswith(("-", "--")) for idx in range(1, len(unknown), 2))
    for idx in range(0, len(unknown), 2):
        arg = unknown[idx]
        # If override one you already have in json
        if arg in new_params:
            idx2 = new_params.index(arg)
            new_params[idx2:idx2+2] = unknown[idx:idx+2]
        # If override one that is in config.py by not in json
        else:
            new_params.extend(unknown[idx:idx+2])
    args = parser.parse_args(new_params)
    top_names = {}
    reconstructed_nested_args(args, top_names, sub_parsers, prefix="")
    # final_args = argparse.Namespace(**top_names)
    final_args = createdBoolDottedDict(top_names)
    # turn_to_dotdicts(final_args)
    return final_args