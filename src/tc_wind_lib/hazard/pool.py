"""
# Stack into ensemble dimension → (source, return_period, lat, lon)
ensemble = stack_rp_maps([rp_map_a, rp_map_b, rp_map_c])

# Summarise across sources — mean, percentiles, spread
mean_map   = ensemble.mean("source")
p10, p90   = ensemble.quantile([0.1, 0.9], dim="source")
spread_map = ensemble.max("source") - ensemble.min("source")

# Slice by metadata — e.g. "what does uncertainty from wind profile look like,
# holding GCM fixed?"
by_profile = ensemble.sel(source=ensemble.attrs_match(gcm="HadGEM3-GC31-HM"))
"""