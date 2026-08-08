"""Create reproducible QGIS study-area layers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from hardinge_high_flow.config import load_config


def _qgis_api() -> dict[str, Any]:
    try:
        from qgis.core import (
            QgsCategorizedSymbolRenderer,
            QgsCoordinateTransformContext,
            QgsFeature,
            QgsField,
            QgsFillSymbol,
            QgsGeometry,
            QgsMarkerSymbol,
            QgsPointXY,
            QgsProject,
            QgsRectangle,
            QgsRendererCategory,
            QgsVectorFileWriter,
            QgsVectorLayer,
        )
        from qgis.PyQt.QtCore import QVariant
    except ImportError as exc:
        raise RuntimeError("Run this script with the QGIS Python interpreter.") from exc

    return {
        "QVariant": QVariant,
        "QgsCategorizedSymbolRenderer": QgsCategorizedSymbolRenderer,
        "QgsCoordinateTransformContext": QgsCoordinateTransformContext,
        "QgsFeature": QgsFeature,
        "QgsField": QgsField,
        "QgsFillSymbol": QgsFillSymbol,
        "QgsGeometry": QgsGeometry,
        "QgsMarkerSymbol": QgsMarkerSymbol,
        "QgsPointXY": QgsPointXY,
        "QgsProject": QgsProject,
        "QgsRectangle": QgsRectangle,
        "QgsRendererCategory": QgsRendererCategory,
        "QgsVectorFileWriter": QgsVectorFileWriter,
        "QgsVectorLayer": QgsVectorLayer,
    }


def _replace_layer(project: Any, layer_name: str) -> None:
    for layer in project.mapLayersByName(layer_name):
        project.removeMapLayer(layer.id())


def _create_point_layer(config: dict[str, Any], api: dict[str, Any]) -> Any:
    layer = api["QgsVectorLayer"](
        "Point?crs=EPSG:4326",
        "Study points",
        "memory",
    )
    if not layer.isValid():
        raise RuntimeError("Could not create the point layer.")

    provider = layer.dataProvider()
    provider.addAttributes(
        [
            api["QgsField"]("name", api["QVariant"].String),
            api["QgsField"]("source", api["QVariant"].String),
            api["QgsField"]("role", api["QVariant"].String),
            api["QgsField"]("longitude", api["QVariant"].Double),
            api["QgsField"]("latitude", api["QVariant"].Double),
        ]
    )
    layer.updateFields()

    features = []
    for point in config["study_area"]["points"]:
        feature = api["QgsFeature"](layer.fields())
        longitude = float(point["longitude"])
        latitude = float(point["latitude"])
        feature.setGeometry(
            api["QgsGeometry"].fromPointXY(api["QgsPointXY"](longitude, latitude))
        )
        feature.setAttributes(
            [
                point["name"],
                point["source"],
                point["role"],
                longitude,
                latitude,
            ]
        )
        features.append(feature)

    if not provider.addFeatures(features):
        raise RuntimeError("Could not add study points.")
    layer.updateExtents()

    palette = config["figures"]["palette"]
    categories = []
    for point in config["study_area"]["points"]:
        color = palette[point["color_key"]]
        symbol = api["QgsMarkerSymbol"].createSimple(
            {
                "name": point["marker"],
                "color": color,
                "outline_color": "#202020",
                "outline_width": "0.35",
                "size": "4.0",
            }
        )
        categories.append(
            api["QgsRendererCategory"](
                point["source"],
                symbol,
                point["name"],
            )
        )
    layer.setRenderer(api["QgsCategorizedSymbolRenderer"]("source", categories))
    return layer


def _create_domain_layer(config: dict[str, Any], api: dict[str, Any]) -> Any:
    layer = api["QgsVectorLayer"](
        "Polygon?crs=EPSG:4326",
        "Data domains",
        "memory",
    )
    if not layer.isValid():
        raise RuntimeError("Could not create the domain layer.")

    provider = layer.dataProvider()
    provider.addAttributes(
        [
            api["QgsField"]("name", api["QVariant"].String),
            api["QgsField"]("source", api["QVariant"].String),
            api["QgsField"]("north", api["QVariant"].Double),
            api["QgsField"]("west", api["QVariant"].Double),
            api["QgsField"]("south", api["QVariant"].Double),
            api["QgsField"]("east", api["QVariant"].Double),
        ]
    )
    layer.updateFields()

    features = []
    for domain in config["study_area"]["domains"]:
        north, west, south, east = map(float, domain["area"])
        feature = api["QgsFeature"](layer.fields())
        feature.setGeometry(
            api["QgsGeometry"].fromRect(api["QgsRectangle"](west, south, east, north))
        )
        feature.setAttributes(
            [domain["name"], domain["source"], north, west, south, east]
        )
        features.append(feature)

    if not provider.addFeatures(features):
        raise RuntimeError("Could not add data domains.")
    layer.updateExtents()

    palette = config["figures"]["palette"]
    categories = []
    for domain in config["study_area"]["domains"]:
        color = palette[domain["color_key"]]
        symbol = api["QgsFillSymbol"].createSimple(
            {
                "color": "255,255,255,0",
                "outline_color": color,
                "outline_style": domain["line_style"],
                "outline_width": "0.8",
            }
        )
        categories.append(
            api["QgsRendererCategory"](
                domain["source"],
                symbol,
                domain["name"],
            )
        )
    layer.setRenderer(api["QgsCategorizedSymbolRenderer"]("source", categories))
    return layer


def _export_geojson(layer: Any, destination: Path, api: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = api["QgsVectorFileWriter"].SaveVectorOptions()
    options.driverName = "GeoJSON"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = api["QgsVectorFileWriter"].CreateOrOverwriteFile
    result = api["QgsVectorFileWriter"].writeAsVectorFormatV3(
        layer,
        str(destination),
        api["QgsCoordinateTransformContext"](),
        options,
    )
    if result[0] != api["QgsVectorFileWriter"].NoError:
        raise RuntimeError(f"Could not export {destination}: {result}")


def create_study_layers(
    config_path: Path,
    export_directory: Path | None = None,
) -> tuple[Any, Any]:
    config = load_config(config_path)
    api = _qgis_api()
    project = api["QgsProject"].instance()

    _replace_layer(project, "Study points")
    _replace_layer(project, "Data domains")

    point_layer = _create_point_layer(config, api)
    domain_layer = _create_domain_layer(config, api)
    project.addMapLayer(domain_layer)
    project.addMapLayer(point_layer)

    if export_directory is not None:
        _export_geojson(
            point_layer,
            export_directory / "study_points.geojson",
            api,
        )
        _export_geojson(
            domain_layer,
            export_directory / "data_domains.geojson",
            api,
        )

    return point_layer, domain_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create configured QGIS study-area layers."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--export-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gis",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Add layers without exporting GeoJSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_directory = None if args.no_export else args.export_directory
    create_study_layers(args.config.resolve(), export_directory)
    print("Created QGIS study points and data domains.")


if __name__ == "__main__":
    main()
