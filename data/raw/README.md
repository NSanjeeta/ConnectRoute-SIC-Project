# Raw data

Raw downloads are immutable inputs and are excluded from version control.

`SRFG-v1.csv` was downloaded from:
https://raw.githubusercontent.com/mherlich/wireless-data-set/main/SRFG-v1.csv

The paper associated with the dataset is Farthofer et al. (2022),
doi:10.1109/OJCOMS.2022.3210289. Check the dataset-specific license and
attribution requirements in the paper/data portal before redistribution.
The project plan identifies the dataset as CC-BY. The source repository's
Apache-2.0 license covers its software; do not assume it also licenses the CSV.
Downloaded CSV SHA-256:
`0DAD68B122E4625FEEEC5506DE80B8F37B73755F5AE33F4BFFA32E40DA3825B7`.

The public ClickHouse tower snapshot was downloaded from:
https://datasets.clickhouse.com/cell_towers.csv.xz

The snapshot contains 43,276,150 rows. Its SHA-256 is
`4A08F7D35AC8A03D1706B0C42988F0C58087148BB11A9A62929D49D8809DA196`.

OpenCelliD country downloads require an account and token and currently cover a
recent observation window. Check its current terms and visible attribution
requirements before using data. Do not store API tokens in this folder.

OpenCelliD attribution and license: https://docs.opencellid.org/docs/attribution
(CC BY-SA 4.0). The global snapshot is retained as a local input; only the
Austria and India LTE subsets are generated as processed data. The India subset
uses MCC 404 and 405; it contains 359,817 valid LTE records, of which 71,823
meet the route demo's minimum sample-count threshold of five.

The Salzburg corridor terrain tile is
`Copernicus_DSM_COG_10_N47_00_E013_00_DEM.tif`, downloaded from the public
Copernicus GLO-30 AWS bucket:
https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E013_00_DEM/Copernicus_DSM_COG_10_N47_00_E013_00_DEM.tif

The tile is approximately 42 MB and contains the N47/E013 one-degree area.
Copernicus DEM GLO-30 Public is a digital surface model (DSM), so it includes
vegetation and buildings rather than bare-earth elevation. It is distributed
under Copernicus DEM terms:
https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM

Tile SHA-256:
`0C5AE70D6D58212F8ECD001C0E1A88E374EA42457A16D8CA0AC971FC2C29C680`.
Accessed 7 November 2026.

Dataset documentation, attribution, and citation guidance:
https://registry.opendata.aws/copernicus-dem/
https://copernicus-dem-30m.s3.amazonaws.com/readme.html

Credit the Copernicus DEM in reports and presentations using the dataset
citation guidance; the map demo should visibly identify the DEM source and
product.

For route terrain profiles, the app reads the required GLO-30 COG tiles directly
from the public AWS bucket and caches only the projected local route-area DEM
under `data/processed/route_dems/`. The first terrain lookup needs internet
access and can take longer than subsequent lookups.
