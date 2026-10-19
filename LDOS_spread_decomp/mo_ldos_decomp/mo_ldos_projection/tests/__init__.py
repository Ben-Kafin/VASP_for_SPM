# -*- coding: utf-8 -*-
"""Test suite for mo_ldos_projection (spec Section 10), built against the
example VASP calculation directory. Run with:

    python -m pytest mo_ldos_projection/tests -v

The example directory is located via the MO_LDOS_EXAMPLE_DIR environment
variable, falling back to the development-layout default
<repo>/MOPM_VASP_Example/wvchg. Tests that need files the example does not
provide (or a CUDA GPU) skip with a clear reason.
"""
