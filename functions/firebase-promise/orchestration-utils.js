class DependencyNotReadyError extends Error {}

/**
 * Create child tasks in the same tasks collection.
 * @param {object} event Firestore event containing after.ref
 * @param {Array<{taskId: string, data: object}>} children
 */
function createChildTasks(event, children) {
  const firestore = event.data.after.ref.parent.firestore;
  const writes = children.map((child) =>
    firestore.doc(`tasks/${child.taskId}`).set(child.data)
  );
  return Promise.all(writes);
}

/**
 * Wrap a task handler to enforce dependencies and mark fulfillment.
 * @param {Function} handler User-defined task function
 * @param {string[]} dependencies Array of regex strings for dependent task IDs
 */
function wrapTaskWithOrchestration(handler, dependencies) {
  return async (event) => {
    const collection = event.data.after.ref.parent;
    const snapshot = await collection.get();
    const statusMap = snapshot.docs.reduce((map, doc) => {
      map[doc.id] = doc.data().status;
      return map;
    }, {});
    const unmet = dependencies.filter((pattern) => {
      const regex = new RegExp(pattern);
      return !Object.entries(statusMap).some(
        ([id, status]) => regex.test(id) && status === "fulfilled"
      );
    });
    if (unmet.length) {
      throw new DependencyNotReadyError(
        `Dependencies not ready: ${unmet.join(", ")}`
      );
    }
    const result = await handler(event);
    await event.data.after.ref.set(
      { status: "fulfilled", ...result },
      { merge: true }
    );
    return result;
  };
}

module.exports = {
  createChildTasks,
  wrapTaskWithOrchestration,
  DependencyNotReadyError,
};
