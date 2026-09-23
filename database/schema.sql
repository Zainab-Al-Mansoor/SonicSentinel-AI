-- SonicSentinel AI database schema (SQLite dialect).
-- Generated from src/models.py. The app creates these tables automatically (db.create_all()).

CREATE TABLE model_versions (
	id INTEGER NOT NULL, 
	model_type VARCHAR(10), 
	version VARCHAR(60), 
	details TEXT, 
	registered_at DATETIME, 
	PRIMARY KEY (id)
);

CREATE TABLE settings (
	"key" VARCHAR(60) NOT NULL, 
	value TEXT, 
	updated_at DATETIME, 
	PRIMARY KEY ("key")
);

CREATE TABLE system_notifications (
	id INTEGER NOT NULL, 
	kind VARCHAR(40), 
	message VARCHAR(400), 
	at DATETIME, 
	is_read BOOLEAN, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_system_notifications_at ON system_notifications (at);

CREATE TABLE users (
	id INTEGER NOT NULL, 
	user_code VARCHAR(20), 
	username VARCHAR(50) NOT NULL, 
	email VARCHAR(120) NOT NULL, 
	full_name VARCHAR(120), 
	phone VARCHAR(30), 
	organization VARCHAR(120), 
	password_hash VARCHAR(255) NOT NULL, 
	role VARCHAR(20) NOT NULL, 
	is_active_flag BOOLEAN, 
	failed_logins INTEGER, 
	locked_until DATETIME, 
	created_at DATETIME, 
	last_login DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (email)
);
CREATE UNIQUE INDEX ix_users_username ON users (username);
CREATE UNIQUE INDEX ix_users_user_code ON users (user_code);

CREATE TABLE audit_logs (
	id INTEGER NOT NULL, 
	user_id INTEGER, 
	username VARCHAR(50), 
	action VARCHAR(40), 
	entity VARCHAR(40), 
	entity_id VARCHAR(60), 
	details TEXT, 
	ip VARCHAR(45), 
	at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE INDEX ix_audit_logs_at ON audit_logs (at);
CREATE INDEX ix_audit_logs_action ON audit_logs (action);

CREATE TABLE live_sessions (
	id INTEGER NOT NULL, 
	session_code VARCHAR(30), 
	user_id INTEGER, 
	started_at DATETIME, 
	ended_at DATETIME, 
	windows INTEGER, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE UNIQUE INDEX ix_live_sessions_session_code ON live_sessions (session_code);

CREATE TABLE audio_events (
	id INTEGER NOT NULL, 
	audio_id VARCHAR(40), 
	user_id INTEGER, 
	source VARCHAR(20), 
	live_session_id INTEGER, 
	actual_class VARCHAR(60), 
	original_filename VARCHAR(255), 
	stored_path VARCHAR(500), 
	format VARCHAR(10), 
	file_size INTEGER, 
	duration FLOAT, 
	sample_rate INTEGER, 
	channels INTEGER, 
	bit_depth INTEGER, 
	uploaded_at DATETIME, 
	sha256 VARCHAR(64), 
	fingerprint TEXT, 
	duplicate_of_id INTEGER, 
	near_duplicate_of_id INTEGER, 
	near_duplicate_score FLOAT, 
	quality_label VARCHAR(20), 
	quality TEXT, 
	noise_level_db FLOAT, 
	waveform_path VARCHAR(500), 
	spectrogram_path VARCHAR(500), 
	python_prediction VARCHAR(60), 
	python_confidence FLOAT, 
	python_scores TEXT, 
	python_margin FLOAT, 
	python_segment_index INTEGER, 
	gtm_prediction VARCHAR(60), 
	gtm_confidence FLOAT, 
	gtm_scores TEXT, 
	gtm_margin FLOAT, 
	gtm_status VARCHAR(20), 
	class_match BOOLEAN, 
	consistency_status VARCHAR(30), 
	confidence_diff FLOAT, 
	combined_scores TEXT, 
	overlap_detected BOOLEAN, 
	overlap_classes TEXT, 
	uncertain BOOLEAN, 
	uncertain_reasons TEXT, 
	repeated_count INTEGER, 
	final_category VARCHAR(60), 
	final_confidence FLOAT, 
	confidence_level VARCHAR(10), 
	severity VARCHAR(20), 
	alert_status VARCHAR(30), 
	recommended_action VARCHAR(300), 
	manual_review_required BOOLEAN, 
	review_reasons TEXT, 
	status VARCHAR(30), 
	decision_trace TEXT, 
	reviewed_category VARCHAR(60), 
	reviewer_id INTEGER, 
	reviewed_at DATETIME, 
	overridden BOOLEAN, 
	python_model_version VARCHAR(60), 
	gtm_model_version VARCHAR(60), 
	processing_ms INTEGER, 
	processed_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	FOREIGN KEY(live_session_id) REFERENCES live_sessions (id), 
	FOREIGN KEY(duplicate_of_id) REFERENCES audio_events (id), 
	FOREIGN KEY(near_duplicate_of_id) REFERENCES audio_events (id), 
	FOREIGN KEY(reviewer_id) REFERENCES users (id)
);
CREATE INDEX ix_audio_events_user_id ON audio_events (user_id);
CREATE INDEX ix_audio_events_uploaded_at ON audio_events (uploaded_at);
CREATE INDEX ix_audio_events_sha256 ON audio_events (sha256);
CREATE UNIQUE INDEX ix_audio_events_audio_id ON audio_events (audio_id);
CREATE INDEX ix_audio_events_severity ON audio_events (severity);
CREATE INDEX ix_audio_events_status ON audio_events (status);
CREATE INDEX ix_audio_events_final_category ON audio_events (final_category);

CREATE TABLE alerts (
	id INTEGER NOT NULL, 
	event_id INTEGER, 
	category VARCHAR(60), 
	severity VARCHAR(20), 
	message VARCHAR(300), 
	recommended_action VARCHAR(300), 
	audience TEXT, 
	status VARCHAR(20), 
	created_at DATETIME, 
	handled_by_id INTEGER, 
	handled_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_id) REFERENCES audio_events (id), 
	FOREIGN KEY(handled_by_id) REFERENCES users (id)
);
CREATE INDEX ix_alerts_created_at ON alerts (created_at);
CREATE INDEX ix_alerts_status ON alerts (status);
CREATE INDEX ix_alerts_event_id ON alerts (event_id);

CREATE TABLE reviews (
	id INTEGER NOT NULL, 
	event_id INTEGER, 
	reviewer_id INTEGER, 
	original_category VARCHAR(60), 
	decided_category VARCHAR(60), 
	decision VARCHAR(20), 
	override BOOLEAN, 
	comment TEXT, 
	recommended_action VARCHAR(300), 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_id) REFERENCES audio_events (id), 
	FOREIGN KEY(reviewer_id) REFERENCES users (id)
);
CREATE INDEX ix_reviews_event_id ON reviews (event_id);

CREATE TABLE segments (
	id INTEGER NOT NULL, 
	event_id INTEGER, 
	"index" INTEGER, 
	start_s FLOAT, 
	end_s FLOAT, 
	audio_path VARCHAR(500), 
	rms_db FLOAT, 
	python_scores TEXT, 
	python_prediction VARCHAR(60), 
	python_confidence FLOAT, 
	gtm_scores TEXT, 
	gtm_prediction VARCHAR(60), 
	gtm_confidence FLOAT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(event_id) REFERENCES audio_events (id)
);
CREATE INDEX ix_segments_event_id ON segments (event_id);

CREATE TABLE alert_actions (
	id INTEGER NOT NULL, 
	alert_id INTEGER, 
	user_id INTEGER, 
	action VARCHAR(20), 
	note VARCHAR(500), 
	at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(alert_id) REFERENCES alerts (id), 
	FOREIGN KEY(user_id) REFERENCES users (id)
);
CREATE INDEX ix_alert_actions_alert_id ON alert_actions (alert_id);
