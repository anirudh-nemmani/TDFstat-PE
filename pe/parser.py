import bilby_pipe.parser
#from bilby_pipe.parser import StoreBoolean 


import argparse

def create_parser(top_level=True):
    """
    Creating and defining a parser for TDFstat arguments
    """
    parser = bilby_pipe.parser.create_parser(top_level=top_level)
    group = parser.add_argument_group("TDFSTAT-PE_arguments", description="PE/Search arguments")

    # Required arguments
    group.add("--indir", type=str, default=None,
                        help="Input data directory (string)")
    group.add("--outdir", type=str, default=None,
                        help="Output directory (string)")
    group.add("--band", type=int, default=None,
                        help="Band number (int)")
    group.add("--seg", type=int, default=None,
                        help="Segment number (int)")
    group.add("--hemi", type=int, default=0,
                        help="Hemisphere [1, 2 or 0 for both] (int)")
    group.add("--thr", type=float, default=None,
                        help="Fstatistic threshold of candidates (double)")
    group.add("--nod", type=int, default=None,
                        help="Length of input time series in days (int)")
    group.add("--dt", type=float, default=None,
                        help="Input time series sampling interval in seconds (double)")
    group.add("--overlap", type=float, default=None,
                        help="Bands overlap; band frequency fpo=10+(1-overlap)*B/(2*dt) (double)")
    group.add("--narrowdown", type=float, default=0.5,
                        help="Output range of f [0-0.5]; if < 0 it is calculated from overlap (double)")
    group.add("--fstat-norm", type=str, default=None,
                        help="Fstatistic normalization (NULL=white noise or blocks_avg)")

    # Optional arguments
    group.add("--usedet", type=str, default=None,
                        help="Use only specified detectors (e.g. H1)")
    group.add("--range-file", type=str, default=None,
                        help="Name of the file with search ranges")
    group.add("--dump-range-file", type=str, default=None,
                        help="Name of the file to dump max. search ranges and exit")
    group.add("--addsig", type=str, default=None,
                        help="Name of the file with signals to be injected")

    # Flag arguments
    group.add("--veto-flag", type=int, default=0, choices=[0, 1],
                        help="Veto lines: 0=no, 1=yes")
    group.add("--gen-vlines-flag", type=int, default=0, choices=[0, 1],
                        help="If 1, generate vlines file and exit")
    group.add("--checkp-flag", type=int, default=0, choices=[0, 1],
                        help="Write checkpoint file on every triggers buffer flush")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)