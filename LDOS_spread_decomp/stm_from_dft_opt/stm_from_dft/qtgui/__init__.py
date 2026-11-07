"""
qtgui: the one shared Qt layer of every interactive window of VASP_for_SPM (PySide6 + pyqtgraph),
the sm4_Browser layer of the conversion guide ("From a matplotlib-wired GUI to a real GUI",
section 9) carried into this suite.

  qtplot.py     figures, axes, artists, events, the toolbar, the information panel, colour maps
                and norms, the text-size rule, the title rule, the cell layout, the publication
                export (import it as `plt`)
  controls.py   the control set (one class per kind, one interface), the shared groups
                (SigmaBox, GammaBox, StepButtons, RunControls) and the layout helpers
  widgets.py    the control table (SPECS, LABELS, SHORT_LABELS) and HoverLabels, the hover label
                of a panel
  mplexport.py  a figure rebuilt in matplotlib for publication files (PDF, EPS, SVG, PNG, TIFF)

This folder is its package's own copy of the layer (the same files are the copy of each of the two
packages that carry one): its package imports it relatively, with no walk up the folders,

    from .qtgui import qtplot as plt, controls, widgets

and its modules import one another relatively (from . import qtplot as plt).
"""
