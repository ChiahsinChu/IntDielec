# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import logging
import os

fname = os.path.join(__path__[0], "../config.json")
if os.path.exists(fname):
    with open(fname, "r") as f:
        CONFIGS = json.loads(f.read())
else:
    CONFIGS = {}

logging.basicConfig(
    filename="eps_cal.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
)
