const { onDocumentWritten } = require("firebase-functions/v2/firestore");
const firebasePromise = require("./firebase-choreographer");
const { _getDbInstance, _getLoggerInstance } = require("./firebase-choreographer");
const {
  wrapTaskWithOrchestration,
  createChildTasks,
  DependencyNotReadyError,
} = require("./firebase-choreographer/utils");

/**
 * Creates a Firestore trigger function with default or overridden path.
 *
 * @param {object} userTaskHandlers
 *   mapping task IDs (exact or prefix) → { handler: Function, dependencies: string[], prefix?: boolean }.
 * @param {object} [options]
 *   { pathPattern?: string, globalFunctionOptions?: {timeoutSeconds, memory, ...} }.
 *   pathPattern overrides default set via initializeChoreographer.
 */
function createTaskWatcher(userTaskHandlers, options = {}) {
  const { pathPattern, globalFunctionOptions = {} } = options;
  const watcherPath = pathPattern || firebasePromise.createTaskWatcher._defaultPathPattern;
  if (!watcherPath) {
    throw new Error("A valid taskPathPattern is required.");
  }
  const logger = _getLoggerInstance();
  if (!watcherPath.includes("{taskId}")) {
    logger.warn(`[TaskWatcher] Path "${watcherPath}" missing {taskId} wildcard.`);
  }

  return onDocumentWritten(
    { document: watcherPath, ...globalFunctionOptions },
    async (event) => {
      const db = _getDbInstance();
      const params = event.params || {};
      const taskId = params.taskId;
      if (!taskId) {
        logger.error(
          "[TaskWatcher] Could not extract taskId from event.params.",
          { params: event.params, pattern: watcherPath }
        );
        return null;
      }
      const change = event.data;
      const beforeData = change.before.data() || {};
      const afterData = change.after.data() || {};
      const status = afterData.status;
      const handlerConfig =
        userTaskHandlers[taskId] ||
        Object.entries(userTaskHandlers).find(
          ([key, cfg]) => cfg.prefix && taskId.startsWith(key)
        )?.[1];

      logger.info(
        `[TaskWatcher] Event for ${taskId} @ ${event.data.after.ref.path}: status=${status}`
      );

      // Processing trigger
      if (status === "processing" && handlerConfig) {
        const orchestrated = wrapTaskWithOrchestration(
          handlerConfig.handler,
          handlerConfig.dependencies || []
        );
        try {
          await orchestrated(event);
          logger.info(`[TaskWatcher] Completed processing ${taskId}`);
        } catch (err) {
          if (err instanceof DependencyNotReadyError) {
            logger.info(`[TaskWatcher] Dependencies not ready for ${taskId}`);
          } else {
            logger.error(`[TaskWatcher] Handler error for ${taskId}:`, err);
          }
        }
      }

      // Re-trigger dependent tasks on fulfillment (only once per transition)
      if (status === "fulfilled" && beforeData.status !== "fulfilled") {
        const collection = event.data.after.ref.parent;
        const allSnap = await collection.get();
        const statuses = allSnap.docs.reduce((acc, doc) => {
          acc[doc.id] = doc.data().status;
          return acc;
        }, {});
        for (const [childKey, childCfg] of Object.entries(userTaskHandlers)) {
          if (!childCfg.dependencies || childCfg.dependencies.length === 0) continue;
          const ready = childCfg.dependencies.every((pattern) =>
            Object.entries(statuses).some(
              ([id, st]) => new RegExp(pattern).test(id) && st === "fulfilled"
            )
          );
          if (ready) {
            const childDoc = collection.doc(childKey);
            const childSnap = await childDoc.get();
            if (!childSnap.exists) {
              await createChildTasks(event, [
                { taskId: childKey, data: { status: "processing" } },
              ]);
              logger.info(`[TaskWatcher] Spawned dependent task ${childKey}`);
            }
          }
        }
      }

      return null;
    }
  );
}

module.exports = { createTaskWatcher };
