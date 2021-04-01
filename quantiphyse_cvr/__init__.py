"""
CVR Quantiphyse plugin

Author: Martin Craig <martin.craig@nottingham.ac.uk>
Copyright (c) 2021 University of Nottingham, Martin Craig
"""
from .widget import CvrPetCo2Widget
from .process import CvrPetCo2GlmProcess, CvrPetCo2VbProcess

QP_MANIFEST = {
    "widgets" : [CvrPetCo2Widget],
    "processes" : [CvrPetCo2GlmProcess, CvrPetCo2VbProcess],
}
