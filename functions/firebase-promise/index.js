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
 * Initializes the Firebase Choreographer package.
 * @param {object} adminInstance - The initialized Firebase Admin SDK instance.
 * @param {object} options - Configuration options:
 *                           { logger?: object, tasksPathPattern: string }.
 */
function initializeChoreographer(adminInstance, options = {}) {
  if (!adminInstance) {
    throw new Error("Firebase Admin instance is required for initializeChoreographer.");
  }
  if (!adminInstance.firestore) {
    throw new Error("Invalid Firebase Admin instance: firestore service not available.");
  }
  const { logger, tasksPathPattern } = options;
  if (!tasksPathPattern) {
    throw new Error("tasksPathPattern is required in initializeChoreographer options.");
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
    throw new Error("Choreographer not initialized. Call initializeChoreographer first.");
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
  initializeChoreographer,
  createTaskWatcher,
  createChildTasks,
  DependencyNotReadyError,
  _getDbInstance,
  _getLoggerInstance,
};
