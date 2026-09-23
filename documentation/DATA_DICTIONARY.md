# Data dictionary – database tables

Generated from `src/models.py`. JSON columns are stored as TEXT.

## `model_versions`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| model_type | VARCHAR(10) | yes |  |
| version | VARCHAR(60) | yes |  |
| details | TEXT | yes |  |
| registered_at | DATETIME | yes |  |

## `settings`

| Column | Type | Null | Key |
|---|---|---|---|
| key | VARCHAR(60) | no | PK |
| value | TEXT | yes |  |
| updated_at | DATETIME | yes |  |

## `system_notifications`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| kind | VARCHAR(40) | yes |  |
| message | VARCHAR(400) | yes |  |
| at | DATETIME | yes | index |
| is_read | BOOLEAN | yes |  |

## `users`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| user_code | VARCHAR(20) | yes | index |
| username | VARCHAR(50) | no | index |
| email | VARCHAR(120) | no |  |
| full_name | VARCHAR(120) | yes |  |
| phone | VARCHAR(30) | yes |  |
| organization | VARCHAR(120) | yes |  |
| password_hash | VARCHAR(255) | no |  |
| role | VARCHAR(20) | no |  |
| is_active_flag | BOOLEAN | yes |  |
| failed_logins | INTEGER | yes |  |
| locked_until | DATETIME | yes |  |
| created_at | DATETIME | yes |  |
| last_login | DATETIME | yes |  |

## `audit_logs`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| user_id | INTEGER | yes | FK → users.id |
| username | VARCHAR(50) | yes |  |
| action | VARCHAR(40) | yes | index |
| entity | VARCHAR(40) | yes |  |
| entity_id | VARCHAR(60) | yes |  |
| details | TEXT | yes |  |
| ip | VARCHAR(45) | yes |  |
| at | DATETIME | yes | index |

## `live_sessions`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| session_code | VARCHAR(30) | yes | index |
| user_id | INTEGER | yes | FK → users.id |
| started_at | DATETIME | yes |  |
| ended_at | DATETIME | yes |  |
| windows | INTEGER | yes |  |

## `audio_events`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| audio_id | VARCHAR(40) | yes | index |
| user_id | INTEGER | yes | FK → users.id |
| source | VARCHAR(20) | yes |  |
| live_session_id | INTEGER | yes | FK → live_sessions.id |
| actual_class | VARCHAR(60) | yes |  |
| original_filename | VARCHAR(255) | yes |  |
| stored_path | VARCHAR(500) | yes |  |
| format | VARCHAR(10) | yes |  |
| file_size | INTEGER | yes |  |
| duration | FLOAT | yes |  |
| sample_rate | INTEGER | yes |  |
| channels | INTEGER | yes |  |
| bit_depth | INTEGER | yes |  |
| uploaded_at | DATETIME | yes | index |
| sha256 | VARCHAR(64) | yes | index |
| fingerprint | TEXT | yes |  |
| duplicate_of_id | INTEGER | yes | FK → audio_events.id |
| near_duplicate_of_id | INTEGER | yes | FK → audio_events.id |
| near_duplicate_score | FLOAT | yes |  |
| quality_label | VARCHAR(20) | yes |  |
| quality | TEXT | yes |  |
| noise_level_db | FLOAT | yes |  |
| waveform_path | VARCHAR(500) | yes |  |
| spectrogram_path | VARCHAR(500) | yes |  |
| python_prediction | VARCHAR(60) | yes |  |
| python_confidence | FLOAT | yes |  |
| python_scores | TEXT | yes |  |
| python_margin | FLOAT | yes |  |
| python_segment_index | INTEGER | yes |  |
| gtm_prediction | VARCHAR(60) | yes |  |
| gtm_confidence | FLOAT | yes |  |
| gtm_scores | TEXT | yes |  |
| gtm_margin | FLOAT | yes |  |
| gtm_status | VARCHAR(20) | yes |  |
| class_match | BOOLEAN | yes |  |
| consistency_status | VARCHAR(30) | yes |  |
| confidence_diff | FLOAT | yes |  |
| combined_scores | TEXT | yes |  |
| overlap_detected | BOOLEAN | yes |  |
| overlap_classes | TEXT | yes |  |
| uncertain | BOOLEAN | yes |  |
| uncertain_reasons | TEXT | yes |  |
| repeated_count | INTEGER | yes |  |
| final_category | VARCHAR(60) | yes | index |
| final_confidence | FLOAT | yes |  |
| confidence_level | VARCHAR(10) | yes |  |
| severity | VARCHAR(20) | yes | index |
| alert_status | VARCHAR(30) | yes |  |
| recommended_action | VARCHAR(300) | yes |  |
| manual_review_required | BOOLEAN | yes |  |
| review_reasons | TEXT | yes |  |
| status | VARCHAR(30) | yes | index |
| decision_trace | TEXT | yes |  |
| reviewed_category | VARCHAR(60) | yes |  |
| reviewer_id | INTEGER | yes | FK → users.id |
| reviewed_at | DATETIME | yes |  |
| overridden | BOOLEAN | yes |  |
| python_model_version | VARCHAR(60) | yes |  |
| gtm_model_version | VARCHAR(60) | yes |  |
| processing_ms | INTEGER | yes |  |
| processed_at | DATETIME | yes |  |

## `alerts`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| event_id | INTEGER | yes | FK → audio_events.id |
| category | VARCHAR(60) | yes |  |
| severity | VARCHAR(20) | yes |  |
| message | VARCHAR(300) | yes |  |
| recommended_action | VARCHAR(300) | yes |  |
| audience | TEXT | yes |  |
| status | VARCHAR(20) | yes | index |
| created_at | DATETIME | yes | index |
| handled_by_id | INTEGER | yes | FK → users.id |
| handled_at | DATETIME | yes |  |

## `reviews`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| event_id | INTEGER | yes | FK → audio_events.id |
| reviewer_id | INTEGER | yes | FK → users.id |
| original_category | VARCHAR(60) | yes |  |
| decided_category | VARCHAR(60) | yes |  |
| decision | VARCHAR(20) | yes |  |
| override | BOOLEAN | yes |  |
| comment | TEXT | yes |  |
| recommended_action | VARCHAR(300) | yes |  |
| created_at | DATETIME | yes |  |

## `segments`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| event_id | INTEGER | yes | FK → audio_events.id |
| index | INTEGER | yes |  |
| start_s | FLOAT | yes |  |
| end_s | FLOAT | yes |  |
| audio_path | VARCHAR(500) | yes |  |
| rms_db | FLOAT | yes |  |
| python_scores | TEXT | yes |  |
| python_prediction | VARCHAR(60) | yes |  |
| python_confidence | FLOAT | yes |  |
| gtm_scores | TEXT | yes |  |
| gtm_prediction | VARCHAR(60) | yes |  |
| gtm_confidence | FLOAT | yes |  |

## `alert_actions`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | no | PK |
| alert_id | INTEGER | yes | FK → alerts.id |
| user_id | INTEGER | yes | FK → users.id |
| action | VARCHAR(20) | yes |  |
| note | VARCHAR(500) | yes |  |
| at | DATETIME | yes |  |
