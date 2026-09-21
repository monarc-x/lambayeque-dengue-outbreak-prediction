"""District centroids and climate nodes for Lambayeque (WGS84).

Climate source: Open-Meteo Historical Weather API (no API key required)
https://open-meteo.com/en/docs/historical-weather-api

Each district is assigned to a nearby climate node used for monthly
precipitation, temperature, and humidity features.
"""

from __future__ import annotations

# ubigeo -> (district, province, lat, lon, climate_node)
DISTRICTS: dict[str, tuple[str, str, float, float, str]] = {
    "140101": ("CHICLAYO", "CHICLAYO", -6.7714, -79.8409, "chiclayo"),
    "140102": ("CHONGOYAPE", "CHICLAYO", -6.6408, -79.3892, "chongoyape"),
    "140103": ("ETEN", "CHICLAYO", -6.9078, -79.8644, "eten"),
    "140104": ("ETEN PUERTO", "CHICLAYO", -6.9267, -79.8661, "eten"),
    "140105": ("JOSE LEONARDO ORTIZ", "CHICLAYO", -6.7630, -79.8330, "chiclayo"),
    "140106": ("LA VICTORIA", "CHICLAYO", -6.7950, -79.8450, "chiclayo"),
    "140107": ("LAGUNAS", "CHICLAYO", -7.0830, -79.5850, "zana"),
    "140108": ("MONSEFU", "CHICLAYO", -6.8775, -79.8720, "eten"),
    "140109": ("NUEVA ARICA", "CHICLAYO", -6.8770, -79.3420, "chongoyape"),
    "140110": ("OYOTUN", "CHICLAYO", -6.8520, -79.3030, "chongoyape"),
    "140111": ("PICSI", "CHICLAYO", -6.7180, -79.7700, "chiclayo"),
    "140112": ("PIMENTEL", "CHICLAYO", -6.8360, -79.9340, "pimentel"),
    "140113": ("REQUE", "CHICLAYO", -6.8650, -79.8190, "chiclayo"),
    "140114": ("SANTA ROSA", "CHICLAYO", -6.8800, -79.9200, "pimentel"),
    "140115": ("SAÑA", "CHICLAYO", -6.9220, -79.5840, "zana"),
    "140116": ("CAYALTI", "CHICLAYO", -6.8920, -79.5620, "zana"),
    "140117": ("PATAPO", "CHICLAYO", -6.7350, -79.6350, "tuman"),
    "140118": ("POMALCA", "CHICLAYO", -6.7700, -79.7780, "chiclayo"),
    "140119": ("PUCALA", "CHICLAYO", -6.7850, -79.6550, "tuman"),
    "140120": ("TUMAN", "CHICLAYO", -6.7510, -79.7040, "tuman"),
    "140201": ("FERREÑAFE", "FERREÑAFE", -6.6390, -79.7880, "ferrenafe"),
    "140202": ("CAÑARIS", "FERREÑAFE", -6.0460, -79.2660, "canaris"),
    "140204": ("MANUEL ANTONIO MESONES MURO", "FERREÑAFE", -6.5800, -79.7500, "ferrenafe"),
    "140205": ("PITIPO", "FERREÑAFE", -6.5660, -79.7310, "ferrenafe"),
    "140206": ("PUEBLO NUEVO", "FERREÑAFE", -6.6400, -79.8550, "ferrenafe"),
    "140301": ("LAMBAYEQUE", "LAMBAYEQUE", -6.7010, -79.9060, "lambayeque"),
    "140302": ("CHOCHOPE", "LAMBAYEQUE", -6.1570, -79.6470, "motupe"),
    "140303": ("ILLIMO", "LAMBAYEQUE", -6.4250, -79.8510, "jayanca"),
    "140304": ("JAYANCA", "LAMBAYEQUE", -6.3920, -79.8230, "jayanca"),
    "140305": ("MOCHUMI", "LAMBAYEQUE", -6.5480, -79.8710, "lambayeque"),
    "140306": ("MORROPE", "LAMBAYEQUE", -6.5900, -80.0150, "morrope"),
    "140307": ("MOTUPE", "LAMBAYEQUE", -6.1530, -79.7140, "motupe"),
    "140308": ("OLMOS", "LAMBAYEQUE", -5.9880, -79.7530, "olmos"),
    "140309": ("PACORA", "LAMBAYEQUE", -6.4280, -79.8390, "jayanca"),
    "140310": ("SALAS", "LAMBAYEQUE", -6.2800, -79.6100, "motupe"),
    "140311": ("SAN JOSE", "LAMBAYEQUE", -6.7680, -79.9680, "lambayeque"),
    "140312": ("TUCUME", "LAMBAYEQUE", -6.5100, -79.8570, "lambayeque"),
}

# Representative climate nodes (lat, lon) for Open-Meteo Archive requests
CLIMATE_NODES: dict[str, tuple[float, float]] = {
    "chiclayo": (-6.7714, -79.8409),
    "pimentel": (-6.8360, -79.9340),
    "eten": (-6.9078, -79.8644),
    "tuman": (-6.7510, -79.7040),
    "zana": (-6.9220, -79.5840),
    "chongoyape": (-6.6408, -79.3892),
    "ferrenafe": (-6.6390, -79.7880),
    "canaris": (-6.0460, -79.2660),
    "lambayeque": (-6.7010, -79.9060),
    "jayanca": (-6.3920, -79.8230),
    "motupe": (-6.1530, -79.7140),
    "olmos": (-5.9880, -79.7530),
    "morrope": (-6.5900, -80.0150),
}
