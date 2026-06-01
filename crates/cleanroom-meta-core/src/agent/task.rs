#[cfg(not(target_arch = "wasm32"))]
use crate::actor::{ActorMessage, CloneableMessage};

pub use cleanroom_meta_protocol::MetaTask;

#[cfg(not(target_arch = "wasm32"))]
impl ActorMessage for MetaTask {}
#[cfg(not(target_arch = "wasm32"))]
impl CloneableMessage for MetaTask {}

#[cfg(test)]
mod tests {
    use super::MetaTask;
    use cleanroom_meta_protocol::ImageMime;
    use serde_json::json;

    #[test]
    fn test_task_creation() {
        let task = MetaTask::new("Test task");

        assert_eq!(task.prompt, "Test task");
        assert!(!task.completed);
        assert!(task.result.is_none());
        assert!(!task.submission_id.is_nil());
    }

    #[test]
    fn test_task_creation_with_string() {
        let task_str = "Another test task".to_string();
        let task = MetaTask::new(task_str);

        assert_eq!(task.prompt, "Another test task");
    }

    #[test]
    fn test_task_serialization() {
        let task = MetaTask::new("Serialize me");

        let serialized = serde_json::to_string(&task).unwrap();
        assert!(serialized.contains("Serialize me"));
        assert!(serialized.contains("submission_id"));
        assert!(serialized.contains("completed"));

        let deserialized: MetaTask = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.prompt, task.prompt);
        assert_eq!(deserialized.submission_id, task.submission_id);
        assert_eq!(deserialized.completed, task.completed);
    }

    #[test]
    fn test_task_with_result() {
        let mut task = MetaTask::new("MetaTask with result");
        let result_value = json!({"output": "success", "value": 42});
        task.result = Some(result_value.clone());
        task.completed = true;

        assert!(task.completed);
        assert_eq!(task.result, Some(result_value));
    }

    #[test]
    fn test_task_unique_submission_ids() {
        let task1 = MetaTask::new("MetaTask 1");
        let task2 = MetaTask::new("MetaTask 2");

        assert_ne!(task1.submission_id, task2.submission_id);
    }

    #[test]
    fn test_task_clone() {
        let original = MetaTask::new("Original task");
        let cloned = original.clone();

        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.submission_id, cloned.submission_id);
        assert_eq!(original.completed, cloned.completed);
        assert_eq!(original.result, cloned.result);
    }

    #[test]
    fn test_task_debug() {
        let task = MetaTask::new("Debug test");
        let debug_str = format!("{task:?}");

        assert!(debug_str.contains("MetaTask"));
        assert!(debug_str.contains("Debug test"));
    }

    #[test]
    fn test_task_with_image() {
        let image_data = vec![0x89, 0x50, 0x4E, 0x47];
        let task = MetaTask::new_with_image("MetaTask with image", ImageMime::PNG, image_data.clone());

        assert_eq!(task.prompt, "MetaTask with image");
        assert!(task.image.is_some());
        if let Some((mime, data)) = &task.image {
            assert_eq!(*mime, ImageMime::PNG);
            assert_eq!(*data, image_data);
        }
        assert!(!task.completed);
        assert!(task.result.is_none());
    }

    #[test]
    fn test_task_without_image() {
        let task = MetaTask::new("MetaTask without image");

        assert_eq!(task.prompt, "MetaTask without image");
        assert!(task.image.is_none());
        assert!(!task.completed);
        assert!(task.result.is_none());
    }

    #[test]
    fn test_task_image_serialization() {
        let image_data = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let task = MetaTask::new_with_image("Serialize with image", ImageMime::JPEG, image_data);

        let serialized = serde_json::to_string(&task).unwrap();
        assert!(serialized.contains("Serialize with image"));
        assert!(serialized.contains("image"));

        let deserialized: MetaTask = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.prompt, task.prompt);
        assert_eq!(deserialized.image, task.image);
        assert_eq!(deserialized.submission_id, task.submission_id);
    }
}
