const {
  wrapTaskWithOrchestration,
  createChildTasks,
  DependencyNotReadyError,
} = require("./orchestration-utils");
 // Lazy-load createTaskWatcher to avoid circular dependency
const createTaskWatcher = (...args) => require("../task-watcher").createTaskWatcher(...args);

let dbInstance;
let packageLoggerInstance = console;

/**
 * Initializes the Firebase Task Manager package.
 * @param {object} adminInstance - The initialized Firebase Admin SDK instance.
 * @param {object} options - Configuration options:
 *                           { logger?: object, tasksPathPattern: string }.
 */
function initializeTaskManager(adminInstance, options = {}) {
  if (!adminInstance) {
    throw new Error("Firebase Admin instance is required for initializeTaskManager.");
  }
  if (!adminInstance.firestore) {
    throw new Error("Invalid Firebase Admin instance: firestore service not available.");
  }
  const { logger, tasksPathPattern } = options;
  if (!tasksPathPattern) {
    throw new Error("tasksPathPattern is required in initializeTaskManager options.");
  }
  dbInstance = adminInstance.firestore();
  packageLoggerInstance = logger || console;
  // Store default path pattern for all watchers
  createTaskWatcher._defaultPathPattern = tasksPathPattern;
}

/**
 * Gets the initialized Firestore instance.
 */
function _getDbInstance() {
  if (!dbInstance) {
    throw new Error("Task manager not initialized. Call initializeTaskManager first.");
  }
  return dbInstance;
}

/**
 * Gets the configured logger instance.
 */
function _getLoggerInstance() {
  return packageLoggerInstance;
}

module.exports = {
  initializeTaskManager,
  createTaskWatcher,
  createChildTasks,
  DependencyNotReadyError,
  _getDbInstance,
  _getLoggerInstance,
};
