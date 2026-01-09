"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import predict, predict_pano, predict_pano_sphere, render


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(predict_pano.predict_pano_cli, "predict-pano")
main_cli.add_command(predict_pano_sphere.predict_pano_sphere_cli, "predict-pano-sphere")
main_cli.add_command(render.render_cli, "render")
