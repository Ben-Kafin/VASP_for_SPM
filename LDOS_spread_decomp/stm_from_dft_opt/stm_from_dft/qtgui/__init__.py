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

Every window finds this package by walking up from its own file to the folder that holds it (the
repository root), the suite's fixed-sibling-path convention:

    import os as _os, sys as _sys
    _root = _os.path.dirname(_os.path.abspath(__file__))
    while not _os.path.isdir(_os.path.join(_root, 'qtgui')) and _os.path.dirname(_root) != _root:
        _root = _os.path.dirname(_root)
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from qtgui import qtplot as plt, controls, widgets
"""
