# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This is the central dispatch of the `dora` command. From there you can
check grid files, launch XPs, check their logs etc, as well
as doing local runs for debugging.
"""
import argparse

from .grid import grid_action
from .info import info_action
from . import inspect as _inspect
from .launch import launch_action
from .log import fatal, setup_logging, simple_log
from .run import run_action
from .share import import_action, export_action
from ._utils import get_dora_config, get_main


def add_submit_rules(parser):
    parser.add_argument("-r", "--retry", action="store_true",
                        help="Retry failed jobs")
    parser.add_argument("-R", "--replace", action="store_true",
                        help="Replace any running job.")
    parser.add_argument("-D", "--replace_done", action="store_true",
                        help="Also resubmit done jobs.")
    parser.add_argument("--no_git_save", action='store_false', dest='git_save', default=None,
                        help="Temporarily deactivate git_save for any scheduled job.")


def add_slurm_config(parser):
    parser.add_argument("-g", "--gpus", type=int, help="Number of gpus.")
    parser.add_argument("-p", "--partition", help="Partition.")
    parser.add_argument("--dev", action="store_const", dest="partition", const="devlab",
                        help="Use dev partition.")
    parser.add_argument("-c", "--comment", help="Comment.")
    parser.add_argument("--constraint", help="Constraint.")


# Shown at the bottom of `dora --help`. Kept short: the per-command help above
# it already lists what each one does, this is only for the choice that is easy
# to get wrong.
_EPILOG = """\
Use `grid` for anything you schedule, even a single job: a grid file is a record
of what you ran, it lives in version control, it deduplicates experiments by
signature, and re-running it monitors or resumes what is already scheduled.
`launch` keeps none of that book keeping. Use `run` for local debugging.

Deprecated commands still work; they are listed last because something above
answers the same question better.

Use `dora <command> --help` for the flags of any command.
"""


def get_parser():
    parser = argparse.ArgumentParser(
        prog="dora",
        description="Easy grid searches for ML.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--package', '-P',
        default=None,
        help='Training module. '
             'You can also set the DORA_PACKAGE env. In last resort, '
             'Dora will look for a package in the current folder with module defined '
             'at --runfile flag.')
    parser.add_argument(
        '--main_module',
        default=None,
        help='Training exec name. '
             'Dora will search for this module to run within the package provided by --package '
             'flag. You can also set DORA_MAIN_MODULE env. Defaults to \'train\' module.')
    parser.add_argument('--verbose', '-v', action='store_true', help="Show debug info.")
    subparsers = parser.add_subparsers(
        title="command", help="Command to execute", required=True, dest='command')
    grid = subparsers.add_parser(
        "grid", help="Schedule and monitor a grid of experiments. The main entry point.")
    add_submit_rules(grid)
    add_slurm_config(grid)
    grid.add_argument("-C", "--cancel", action='store_true',
                      help="Cancel all running jobs.")
    grid.add_argument("--clear", action='store_true',
                      help="Remove XP folder, reschedule all jobs, starting from scratch.")
    grid.add_argument("-i", "--interval", default=5, type=float,
                      help="Update status and metrics every that number of minutes. "
                           "Default is 5 min.")
    grid.add_argument("--no_monitoring", action="store_false", dest="monitor",
                      help="No monitoring, just schedule and print current state.")

    grid.add_argument("--dry_run", action="store_true",
                      help="Only simulate actions but does not run any call to Slurm.")
    grid.add_argument("-T", "--trim", type=int,
                      help="Trim history to the length of the exp with the given index.")
    grid.add_argument("-L", "--trim_last", action="store_true",
                      help="Trim history to the slowest.")

    group = grid.add_mutually_exclusive_group()
    group.add_argument("-f", "--folder", type=int,
                       help="Show the folder for the job with the given index")
    group.add_argument("-l", "--log", type=int,
                       help="Show the log for the job with the given index")
    group.add_argument("-t", "--tail", type=int,
                       help="Show the log for the job with the given index")

    grid.add_argument("--init", action='store_true',
                      help="Init the given XPs so that their signature can be referenced.")

    grid.add_argument(
        'grid', nargs='?',
        help='Name of the grid to run. Name of the module will be `package`.grids.`name`.')

    grid.add_argument("patterns", nargs='*',
                      help="Only handle experiments matching all the given pattern. "
                           "If empty, handle all experiments")
    grid.set_defaults(action=grid_action)

    run = subparsers.add_parser(
        "run", help="Run one experiment locally, for debugging.")
    run.add_argument("-f", "--from_sig", help="Signature of job to use as baseline.")
    run.add_argument("-d", "--ddp", action="store_true", help="Distributed training.")
    run.add_argument("--ddp_workers", type=int,
                     help="Nb of workers for distributed, default to nb of GPUs.")
    run.add_argument("--git_save", action="store_true", default=False,
                     help="Run from a clean git clone.")
    run.add_argument("--clear", action='store_true',
                     help="Remove XP folder, reschedule job, starting from scratch.")
    run.add_argument("argv", nargs='*')
    run.set_defaults(action=run_action)

    # Read-only inspection. These never import the training package when a
    # dora.toml supplies the experiment directory, and their output is capped
    # and uncoloured so it is cheap to read programmatically.
    def add_inspect(name, help_text, targets_help):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("targets", nargs="+", help=targets_help)
        sub.add_argument("--json", action="store_true",
                         help="Emit one compact JSON object instead of a table.")
        sub.add_argument("--limit", type=int, default=None,
                         help="Maximum rows or lines to show.")
        sub.set_defaults(read_only=True)
        return sub

    status = add_inspect(
        "status", "Compact state of experiments or a whole grid.",
        "Signatures, grid names or Slurm job ids. Prefix with @ to force a signature.")
    status.add_argument("--keys", default=None,
                        help="Comma separated metrics to show instead of the defaults.")
    status.set_defaults(action=_inspect.status_action)

    metrics = add_inspect(
        "metrics", "Downsampled metric history for one experiment.", "A signature.")
    metrics.add_argument("--stage", default=None, help="Stage, e.g. train or valid.")
    metrics.add_argument("--keys", default=None, help="Comma separated metrics to show.")
    metrics.add_argument("--every", type=int, default=None,
                         help="Keep one epoch out of every N, across the whole run.")
    metrics.set_defaults(action=_inspect.metrics_action)

    log = add_inspect("log", "Tail an experiment's log, stripped of colour.",
                      "A signature, grid name or job id.")
    log.add_argument("--tail", type=int, default=None, dest="limit",
                     help="Number of lines to show (same as --limit).")
    log.add_argument("--grep", default=None, help="Only lines matching this regexp.")
    log.add_argument("--rank", type=int, default=None, help="Restrict to one rank.")
    log.add_argument("--job", default=None, help="Look at this job id's logs.")
    log.set_defaults(action=_inspect.log_action)

    why = add_inspect("why", "Explain why an experiment failed.",
                      "A signature, grid name or job id.")
    why.add_argument("--job", default=None, help="Look only at this job id's logs.")
    why.add_argument("--attempts", type=int, default=3,
                     help="How many job attempts to look back through (default 3).")
    why.set_defaults(action=_inspect.why_action)

    plan = subparsers.add_parser(
        "plan", help="Resolve a grid to its experiments without scheduling anything.")
    plan.add_argument("grid", help="Grid name, as for `dora grid`.")
    plan.add_argument("patterns", nargs="*", help="Only experiments matching these.")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--limit", type=int, default=None)
    plan.set_defaults(action=_inspect.plan_action)

    # Superseded, kept working for existing scripts. Listed last and marked as
    # such because each has a better answer above: `grid` instead of `launch`,
    # and `status`/`metrics`/`log`/`why` instead of `info`.
    launch = subparsers.add_parser(
        "launch",
        help="(deprecated) Schedule a single job on Slurm. Use `grid` instead.")
    launch.add_argument("-f", "--from_sig", help="Signature of job to use as baseline.")
    launch.add_argument("-a", "--attach", action="store_true",
                        help="Attach to the remote process. Interrupting the command will "
                             "kill the remote job.")
    launch.add_argument("--no_tail", action="store_false", dest="tail", default=True,
                        help="Does not tail the log once job is started.")
    launch.add_argument("-C", "--cancel", action='store_true',
                        help="Cancel any existing job and return.")
    launch.add_argument("--clear", action='store_true',
                        help="Remove XP folder, reschedule job, starting from scratch.")
    add_submit_rules(launch)
    add_slurm_config(launch)
    launch.add_argument("argv", nargs='*')
    launch.set_defaults(action=launch_action)

    info = subparsers.add_parser(
        "info",
        help="(deprecated) Everything known about one experiment, verbosely. "
             "Use `status`, `metrics`, `log` or `why`.")
    info.add_argument("-f", "--from_sig", help="Signature of job to use as baseline.")
    info.add_argument("-j", "--job_id", help="Find job by job id.")
    info.add_argument("-C", "--cancel", action="store_true", help="Cancel job")
    info.add_argument("-l", "--log", action="store_true", help="Show entire log")
    info.add_argument("-t", "--tail", action="store_true", help="Tail log")
    info.add_argument("-m", "--metrics", action="store_true", help="Show last metrics")
    info.add_argument("argv", nargs='*')
    info.set_defaults(action=info_action)

    import_ = subparsers.add_parser(
        "import",
        help="(deprecated) Read an exported blob on stdin and register "
             "those experiments locally, so their signatures resolve.")
    import_.set_defaults(action=import_action)

    export = subparsers.add_parser(
        "export",
        help="(deprecated) Print a shareable blob describing the given experiments.")
    export.add_argument("sigs", nargs='*', help='All the XP sigs to export.')
    export.set_defaults(action=export_action)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.action is None:
        fatal("You must give an action.")

    if getattr(args, "read_only", False):
        # These only need to know where experiments live. Importing the training
        # package to find that out costs seconds on a real project, so use
        # dora.toml when it can answer, and say so when it cannot.
        dora = get_dora_config()
        if dora is None:
            simple_log("Dora", "No dora.toml with a resolvable `dir`; "
                               "importing the training package to find it "
                               "(this is the slow path).")
            dora = get_main(args.main_module, args.package).dora
        return args.action(args, dora)

    main = get_main(args.main_module, args.package)

    if getattr(args, 'from_sig', None) is not None:
        try:
            argv = main.get_argv_from_sig(args.from_sig)
        except RuntimeError:
            fatal(f"Could not find an existing run with sig {args.from_sig}")
        simple_log("Parser", "Injecting argv", argv, "from sig", args.from_sig)
        args.argv = argv + args.argv

    if getattr(args, 'git_save', None) is not None:
        main.dora.git_save = args.git_save
    args.action(args, main)


if __name__ == "__main__":
    main()
