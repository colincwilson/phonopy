# Container for global variables in string environment.

epsilon = "ϵ"  # avoid "ε" (confusable with IPA), alt. "𝜀"
bos = "⋊"  # beginning of string / start token; alt. "⊢", ">", <s>, <bos>
eos = "⋉"  # end of string / end token; alt. "⊣", "<", </s>, <eos>


def init(param):
    """ Set globals with dictionary or module """
    global epsilon, bos, eos
    if not isinstance(param, dict):
        param = vars(param)
    if "epsilon" in param:
        epsilon = param["epsilon"]
    if "bos" in param:
        bos = param["bos"]
    if "eos" in param:
        eos = param["eos"]
