"""
qtgui: this package's own copy of the Qt layer of VASP_for_SPM's interactive windows (PySide6 + pyqtgraph),
the sm4_Browser layer of the conversion guide ("From a matplotlib-wired GUI to a real GUI",
section 9) carried into this suite.

  qtplot.py     figures, axes, artists, events, the toolbar, the information panel, colour maps
                and norms, the text-size rule, the title rule, the cell layout, the publication
                export (import it as `plt`)
  controls.py   the control set (one class per kind, one interface), the shared groups
                (SigmaBox, GammaBox, StepButtons, RunControls) and the layout helpers
  widgets.py    the control table (SPECS, LABELS, SHORT_LABELS) and HoverLabels, the hover label
                of a panel
  mplexport.py  a figure rebuilt in matplotlib for publication files (PDF, EPS, editable SVG;
                the PNG and TIFF are the window image)

This folder is its package's own copy of the layer (the same files are the copy in each package
that carries one), imported relatively from inside the package:

    from .qtgui import qtplot as plt, controls, widgets
"""
