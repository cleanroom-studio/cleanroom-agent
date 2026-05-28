-- ============================================
-- Cleanroom Agent - Initial Schema
-- Version: 001
-- ============================================

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================
-- 1. S.DEF 文档注册表
-- ============================================
CREATE TABLE sdef_documents (
    name TEXT PRIMARY KEY,
    version TEXT,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 2. 运行时管理表
-- ============================================

-- 2.1 任务表
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL CHECK (task_type IN (
        'REPO_ANALYZE', 'EXTRACT_METADATA', 'EXTRACT_ARCHITECTURE',
        'EXTRACT_DATA_MODEL', 'EXTRACT_MODULE', 'EXTRACT_UI',
        'EXTRACT_TESTS', 'INFER_DESIGN_DECISIONS', 'VALIDATE_SHARD',
        'GENERATE_CODE', 'RUN_TESTS', 'MERGE_CODE', 'IMPORT_SDEF', 'EXPORT_SDEF'
    )),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'assigned', 'in_progress', 'completed',
                          'failed', 'retrying', 'failed_permanently')),
    priority INTEGER NOT NULL DEFAULT 5,
    input_json TEXT NOT NULL,
    output_json TEXT,
    error_message TEXT,
    assigned_to TEXT,
    progress REAL NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    last_heartbeat TIMESTAMP,
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_tasks_status_priority ON tasks(status, priority DESC, created_at);
CREATE INDEX idx_tasks_status_heartbeat ON tasks(status, last_heartbeat);
CREATE INDEX idx_tasks_assigned_to ON tasks(assigned_to);
CREATE INDEX idx_tasks_type_status ON tasks(task_type, status);

-- 2.2 分片状态表
CREATE TABLE shards (
    shard_id TEXT PRIMARY KEY,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    sdef_uri TEXT NOT NULL,
    section_type TEXT NOT NULL,
    file_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'generating', 'generated',
                          'validating', 'validated', 'code_generated', 'tested', 'failed')),
    content_hash TEXT,
    token_estimate INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_shards_status ON shards(status);
CREATE INDEX idx_shards_document ON shards(document_name);
CREATE UNIQUE INDEX idx_shards_uri ON shards(sdef_uri);
CREATE INDEX idx_shards_type_status ON shards(section_type, status);

-- 2.3 智能体注册表
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL CHECK (agent_type IN ('producer', 'consumer', 'orchestrator')),
    capabilities_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'offline'
        CHECK (status IN ('online', 'offline', 'busy')),
    current_task_id TEXT,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2.4 全局命名服务表（符号注册表）
CREATE TABLE symbol_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    sdef_uri TEXT NOT NULL,
    language TEXT NOT NULL,
    symbol_type TEXT NOT NULL
        CHECK (symbol_type IN ('class', 'interface', 'function', 'variable', 'constant', 'enum', 'type')),
    concrete_name TEXT NOT NULL,
    is_user_defined BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_name, language, sdef_uri, symbol_type),
    UNIQUE(document_name, language, concrete_name)
);

CREATE INDEX idx_symbol_registry_uri ON symbol_registry(document_name, language, sdef_uri);

-- 2.5 一致性指纹表
CREATE TABLE fingerprints (
    entity_uri TEXT NOT NULL,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    sdef_hash TEXT,
    db_hash TEXT,
    code_hash TEXT,
    code_path TEXT,
    last_checked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_consistent_at TIMESTAMP,
    PRIMARY KEY (document_name, entity_uri)
);

CREATE INDEX idx_fingerprints_inconsistent ON fingerprints(document_name)
    WHERE sdef_hash != db_hash OR db_hash != code_hash OR sdef_hash != code_hash;

-- 2.6 审计日志表
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    old_value_json TEXT,
    new_value_json TEXT
);

CREATE INDEX idx_audit_log_time ON audit_log(timestamp);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_log_resource_time ON audit_log(resource_type, resource_id, timestamp);

-- 2.7 检查点表
CREATE TABLE checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    task_snapshot_json TEXT NOT NULL,
    shard_snapshot_json TEXT NOT NULL
);

CREATE INDEX idx_checkpoints_document ON checkpoints(document_name, created_at DESC);

-- 2.8 两阶段提交预备表
CREATE TABLE prepared_transactions (
    transaction_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL CHECK (phase IN ('prepare', 'commit', 'rollback')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prepared_at TIMESTAMP,
    committed_at TIMESTAMP,
    rollback_at TIMESTAMP,
    changes_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'prepared', 'committed', 'rolled_back', 'failed'))
);

-- ============================================
-- 3. S.DEF 存储表（分层映射）
-- ============================================

-- 3.1 数据模型层
CREATE TABLE data_models (
    entity TEXT NOT NULL,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'legacy')),
    version TEXT,
    description TEXT,
    logical_model TEXT,
    PRIMARY KEY (document_name, entity)
);

CREATE TABLE data_attributes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    entity TEXT NOT NULL,
    name TEXT NOT NULL,
    attr_type TEXT NOT NULL,
    format TEXT,
    description TEXT,
    required BOOLEAN NOT NULL DEFAULT FALSE,
    identity BOOLEAN NOT NULL DEFAULT FALSE,
    generated BOOLEAN NOT NULL DEFAULT FALSE,
    unique_flag BOOLEAN NOT NULL DEFAULT FALSE,
    internal BOOLEAN NOT NULL DEFAULT FALSE,
    deprecated BOOLEAN NOT NULL DEFAULT FALSE,
    default_value TEXT,
    constraints_json TEXT,
    FOREIGN KEY (document_name, entity) REFERENCES data_models(document_name, entity) ON DELETE CASCADE
);

CREATE TABLE data_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    entity TEXT NOT NULL,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    foreign_key TEXT,
    join_table TEXT,
    on_delete TEXT,
    FOREIGN KEY (document_name, entity) REFERENCES data_models(document_name, entity) ON DELETE CASCADE
);

CREATE INDEX idx_data_attributes_type ON data_attributes(attr_type);

-- 3.2 契约层
CREATE TABLE contracts (
    name TEXT NOT NULL,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    contract_type TEXT NOT NULL CHECK (contract_type IN ('interface', 'class', 'enum', 'api')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'legacy')),
    version TEXT,
    is_abstract BOOLEAN NOT NULL DEFAULT FALSE,
    description TEXT,
    implements_json TEXT,
    dependencies_json TEXT,
    invariants_json TEXT,
    http_method TEXT,
    api_path TEXT,
    auth TEXT,
    rate_limit TEXT,
    deprecated_json TEXT,
    compatibility_json TEXT,
    PRIMARY KEY (document_name, name)
);

CREATE TABLE contract_methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    contract_name TEXT NOT NULL,
    signature TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    behavior TEXT,
    preconditions_json TEXT,
    postconditions_json TEXT,
    errors_json TEXT,
    deprecated_json TEXT,
    FOREIGN KEY (document_name, contract_name) REFERENCES contracts(document_name, name) ON DELETE CASCADE
);

CREATE INDEX idx_contract_methods_contract ON contract_methods(document_name, contract_name);

-- 3.3 行为层
CREATE TABLE function_specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    logic TEXT,
    complexity TEXT,
    pure_function BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE function_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_id INTEGER NOT NULL REFERENCES function_specs(id) ON DELETE CASCADE,
    param_direction TEXT NOT NULL CHECK (param_direction IN ('input', 'output')),
    name TEXT NOT NULL,
    param_type TEXT NOT NULL,
    description TEXT
);

-- 3.4 UI 层
CREATE TABLE ui_documents (
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    pen_version TEXT,
    raw_content_json TEXT NOT NULL,
    PRIMARY KEY (document_name)
);

CREATE TABLE ui_screens (
    id TEXT NOT NULL,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    name TEXT NOT NULL,
    route TEXT,
    purpose TEXT,
    layout_description TEXT,
    PRIMARY KEY (document_name, id)
);

-- 3.5 测试契约层
CREATE TABLE test_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    module_id TEXT,
    interface_id TEXT
);

CREATE TABLE test_cases (
    id TEXT NOT NULL,
    group_id INTEGER NOT NULL REFERENCES test_groups(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    test_type TEXT NOT NULL CHECK (test_type IN ('unit', 'integration')),
    given_json TEXT,
    when_condition TEXT,
    then_expected_json TEXT,
    expected_exception TEXT,
    expected_side_effects_json TEXT,
    steps_json TEXT,
    PRIMARY KEY (group_id, id)
);

-- 3.6 其他层
CREATE TABLE system_boundaries (
    document_name TEXT PRIMARY KEY REFERENCES sdef_documents(name) ON DELETE CASCADE,
    core_purpose TEXT NOT NULL,
    data_json TEXT NOT NULL
);

CREATE TABLE design_decisions (
    id TEXT NOT NULL,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    context TEXT,
    alternatives_json TEXT,
    consequences_json TEXT,
    constraints_json TEXT,
    PRIMARY KEY (document_name, id)
);

CREATE TABLE version_records (
    version TEXT NOT NULL,
    document_name TEXT NOT NULL REFERENCES sdef_documents(name) ON DELETE CASCADE,
    release_date TEXT,
    deprecated BOOLEAN NOT NULL DEFAULT FALSE,
    eol_date TEXT,
    breaking_changes_json TEXT,
    compatibility_notes TEXT,
    PRIMARY KEY (document_name, version)
);

-- ============================================
-- 4. 触发器
-- ============================================

-- 4.1 状态转换验证
CREATE TRIGGER validate_task_status_transition
BEFORE UPDATE OF status ON tasks
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN OLD.status = 'completed' AND NEW.status != 'completed' THEN
            RAISE(ABORT, 'Cannot change status from completed')
        WHEN OLD.status = 'failed_permanently' AND NEW.status NOT IN ('failed_permanently', 'retrying') THEN
            RAISE(ABORT, 'Cannot change status from failed_permanently')
    END;
END;

-- 4.2 进度验证
CREATE TRIGGER validate_task_progress
BEFORE UPDATE OF progress ON tasks
WHEN NEW.progress < OLD.progress
BEGIN
    SELECT RAISE(ABORT, 'Progress cannot decrease');
END;

-- 4.3 自动审计日志 - 任务变更
CREATE TRIGGER log_task_changes
AFTER UPDATE ON tasks
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (actor, action, resource_type, resource_id, old_value_json, new_value_json)
    VALUES (
        COALESCE(NEW.assigned_to, 'system'),
        'update',
        'task',
        NEW.task_id,
        json_object('status', OLD.status, 'progress', OLD.progress),
        json_object('status', NEW.status, 'progress', NEW.progress)
    );
END;

-- 4.4 自动审计日志 - 分片变更
CREATE TRIGGER log_shard_changes
AFTER UPDATE ON shards
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (actor, action, resource_type, resource_id, old_value_json, new_value_json)
    VALUES (
        'system',
        'update',
        'shard',
        NEW.shard_id,
        json_object('status', OLD.status),
        json_object('status', NEW.status)
    );
END;

-- ============================================
-- 5. FTS5 全文搜索
-- ============================================
CREATE VIRTUAL TABLE sdef_fts USING fts5(
    document_name,
    entity_name,
    description,
    content='sdef_documents',
    content_rowid='rowid'
);

-- FTS 同步触发器
CREATE TRIGGER sdef_fts_insert AFTER INSERT ON sdef_documents BEGIN
    INSERT INTO sdef_fts(rowid, document_name, entity_name, description)
    VALUES (NEW.rowid, NEW.name, NEW.name, NEW.description);
END;

CREATE TRIGGER sdef_fts_delete AFTER DELETE ON sdef_documents BEGIN
    INSERT INTO sdef_fts(sdef_fts, rowid, document_name, entity_name, description)
    VALUES ('delete', OLD.rowid, OLD.name, OLD.name, OLD.description);
END;

CREATE TRIGGER sdef_fts_update AFTER UPDATE ON sdef_documents BEGIN
    INSERT INTO sdef_fts(sdef_fts, rowid, document_name, entity_name, description)
    VALUES ('delete', OLD.rowid, OLD.name, OLD.name, OLD.description);
    INSERT INTO sdef_fts(rowid, document_name, entity_name, description)
    VALUES (NEW.rowid, NEW.name, NEW.name, NEW.description);
END;

-- ============================================
-- 6. 迁移版本记录
-- ============================================
CREATE TABLE schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);