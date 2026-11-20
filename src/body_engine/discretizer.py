from schemas import Exteroception, SimConfig


def discretize_exteroception(ext: Exteroception, cfg: SimConfig) -> tuple[str, ...]:
    tags: list[str] = []
    if ext.lighting_lux >= cfg.lighting_harsh_threshold:
        tags.append("harsh_light")
    elif ext.lighting_lux <= cfg.lighting_dim_threshold:
        tags.append("dim_light")
    if ext.ambient_db >= cfg.ambient_alarm_threshold:
        tags.append("alarm_load")
    elif ext.ambient_db >= cfg.ambient_loud_threshold:
        tags.append("loud_room")
    if ext.vocal_tone in {"sharp", "authoritative", "hostile"}:
        tags.append(f"voice_{ext.vocal_tone}")
    if ext.interpersonal_distance_m <= cfg.close_distance_threshold:
        tags.append("crowded")
    if len(ext.active_actors) >= cfg.many_actors_threshold:
        tags.append("many_actors")
    return tuple(tags)
