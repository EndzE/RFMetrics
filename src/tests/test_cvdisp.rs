use super::*;
use crate::metrics::ffvship::CustomDisplay;

fn numbered_display() -> serde_json::Map<String, serde_json::Value> {
    serde_json::from_str(
        r#"{"colorspace":"SDR","resolution":[1920,1080],"viewing_distance_meters":0.6,
            "diagonal_size_inches":24,"max_luminance":200,"contrast":1000,
            "E_ambient":250,"k_refl":0.005,"exposure":1.0,
            "name":"Numbered","source":"rfmetrics-custom"}"#,
    )
    .unwrap()
}

/// Draft round-trips the stored values (resolution stays integer-typed,
/// reflectivity crosses the percent boundary and back).
#[test]
fn draft_round_trip() {
    let map = numbered_display();
    let back = DisplayDraft::from_map("Numbered", &map).into_map();
    // Numeric fidelity (representation may normalize int<->float).
    for f in [
        "viewing_distance_meters",
        "diagonal_size_inches",
        "max_luminance",
        "contrast",
        "E_ambient",
        "k_refl",
        "exposure",
    ] {
        assert_eq!(
            back.get(f).and_then(|v| v.as_f64()),
            map.get(f).and_then(|v| v.as_f64()),
            "{f}"
        );
    }
    assert_eq!(back.get("resolution"), map.get("resolution"));
    assert_eq!(back.get("colorspace"), map.get("colorspace"));
}

/// Untouched decimals survive the box precision: the percent round-trip
/// (fraction -> % -> fraction) keeps the exact original instead of
/// drifting by a float ulp.
#[test]
fn preserve_keeps_untouched() {
    // Untouched decimals survive the box precision: the percent
    // round-trip (fraction -> % -> fraction) keeps the exact original
    // instead of drifting by a float ulp.
    let map = numbered_display();
    let draft = DisplayDraft::from_map("Numbered", &map);
    let mut built = draft.clone().into_map();
    preserve(
        &mut built,
        "k_refl",
        draft.reflect_pct,
        0.005 * 100.0,
        0.005,
        2,
    );
    assert_eq!(built.get("k_refl"), map.get("k_refl"));
    // Edited value sticks instead (epsilon: division paths differ by ulps).
    let mut edited = draft.clone();
    edited.reflect_pct = 0.6;
    let mut built2 = edited.into_map();
    preserve(&mut built2, "k_refl", 0.6, 0.005 * 100.0, 0.005, 2);
    let got = built2.get("k_refl").and_then(|v| v.as_f64()).unwrap();
    assert!((got - 0.006).abs() < 1e-12, "{got}");
}

/// Save flows: add, overwrite-own, rename, duplicate refusals.
#[test]
fn save_flows() {
    let map = numbered_display();
    let mut list = Vec::new();
    // Add: stored, name returned.
    assert_eq!(
        save_custom(&mut list, " Mine ", map.clone(), None),
        Ok("Mine".to_owned())
    );
    assert_eq!(list.len(), 1);
    // Duplicate (own list or registry) refused…
    assert!(save_custom(&mut list, "Mine", map.clone(), None).is_err());
    assert!(save_custom(&mut list, "standard_fhd", map.clone(), None).is_err());
    assert!(save_custom(&mut list, "", map.clone(), None).is_err());
    // …except renaming-own exception.
    assert_eq!(
        save_custom(&mut list, "Mine", map.clone(), Some("Mine")),
        Ok("Mine".to_owned())
    );
    assert_eq!(list.len(), 1);
    // Rename: old gone, new present, selection follows the return.
    assert_eq!(
        save_custom(&mut list, "Ours", map.clone(), Some("Mine")),
        Ok("Ours".to_owned())
    );
    assert_eq!(
        list.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["Ours"]
    );
}

/// Delete removes by name and reports; unknown names are no-ops.
#[test]
fn delete_flows() {
    let mut list = vec![CustomDisplay {
        name: "Mine".to_owned(),
        display: numbered_display(),
    }];
    assert!(!delete_custom(&mut list, "Nobody"));
    assert!(delete_custom(&mut list, "Mine"));
    assert!(list.is_empty());
}
