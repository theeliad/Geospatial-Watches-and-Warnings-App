# state_finder.py

import pandas as pd
from math import radians, cos, sin, asin, sqrt

US_STATE_CENTROIDS = pd.DataFrame({
    "state": [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
        "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
        "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
        "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ],
    "latitude": [
        32.806671, 61.370716, 33.729759, 34.969704, 36.116203, 39.059811, 41.597782, 39.318523,
        27.766279, 33.040619, 21.094318, 44.240459, 40.349457, 39.849426, 42.011539, 38.5266,
        37.66814, 31.169546, 44.693947, 39.063946, 42.230171, 43.326618, 45.694454, 32.741646,
        38.456085, 46.921925, 41.12537, 38.313515, 43.452492, 40.298904, 34.840515, 42.165726,
        35.630066, 47.528912, 40.388783, 35.565342, 44.572021, 40.590752, 41.680893, 33.856892,
        44.299782, 35.747845, 31.054487, 39.32098, 44.045876, 37.769337, 47.400902, 38.491226,
        44.268543, 42.755966
    ],
    "longitude": [
        -86.79113, -152.404419, -111.431221, -92.373123, -119.681564, -105.311104, -72.755371, -75.507141,
        -81.686783, -83.643074, -157.498337, -114.478828, -88.986137, -86.258278, -93.210526, -96.726486,
        -84.670067, -91.867805, -69.381927, -76.802101, -71.530106, -84.536095, -93.900192, -89.678696,
        -92.288368, -110.454353, -98.268082, -117.055374, -71.563896, -74.521011, -106.248482, -74.948051,
        -79.806419, -99.784012, -82.764915, -96.928917, -122.070938, -77.209755, -71.51178, -80.945007,
        -99.438828, -86.692345, -97.563461, -111.093735, -72.710686, -78.169968, -121.490494, -80.954456,
        -89.616508, -107.30249
    ]
})


def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two coordinates."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return 3956 * c  # miles


def find_nearest_state(lat, lon):
    """Finds the U.S. state whose centroid is geographically closest."""
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    try:
        lat, lon = float(lat), float(lon)
        US_STATE_CENTROIDS['distance'] = US_STATE_CENTROIDS.apply(
            lambda row: haversine(lat, lon, row['latitude'], row['longitude']), axis=1
        )
        return US_STATE_CENTROIDS.loc[US_STATE_CENTROIDS['distance'].idxmin(), 'state']
    except Exception:
        return "Unknown"
