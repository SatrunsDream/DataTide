Surf spot list (required for the map and lists)
================================================
Generate from station metadata:

  python scripts/build_surf_spots_json.py

Output: site/public/data/surf_spots.json

Optional legacy model export (not used by the current surf UI):

  python scripts/export_forecast_to_site_json.py ...
