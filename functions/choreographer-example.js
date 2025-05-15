const admin = require("firebase-admin");
const { logger } = require("firebase-functions");
const {
  initializeChoreographer,
  createTaskWatcher,
  createChildTasks,
  DependencyNotReadyError,
} = require("./firebase-choreographer");

// Initialize Admin & Choreographer
if (!admin.apps.length) admin.initializeApp();
initializeChoreographer(admin, { logger, tasksPathPattern: "tasks/{taskId}" });

// Task A: spawns B & C
async function taskA(event) {
  await createChildTasks(event, [
    { taskId: "B", data: { status: "processing" } },
    { taskId: "C", data: { status: "processing" } },
  ]);
  return { message: "A done, spawned B & C", childTaskIds: ["B", "C"] };
}

// Task B logic
async function taskB(event) {
  // ... do work for B ...
  return { message: "B done" };
}

// Task C logic
async function taskC(event) {
  // ... do work for C ...
  return { message: "C done" };
}

// Task D waits on B & C
async function taskD(event) {
  return { message: "D running after B & C" };
}

const userTaskHandlers = {
  A: { handler: taskA, dependencies: [], prefix: false },
  B: { handler: taskB, dependencies: ["^A$"], prefix: false },
  C: { handler: taskC, dependencies: ["^A$"], prefix: false },
  D: { handler: taskD, dependencies: ["^B$", "^C$"], prefix: false },
};

 // Export the single Cloud Function
 exports.choreographerExample = createTaskWatcher(userTaskHandlers);
 
 // HTTP function to start a task via an HTTPS trigger
 exports.startTask = require("firebase-functions").https.onRequest(
   async (req, res) => {
     const { taskId } = req.body;
     if (!taskId) {
       res.status(400).send("Missing taskId");
       return;
     }
     const db = require("firebase-admin").firestore();
     try {
       await db.doc(`tasks/${taskId}`).set({ status: "processing" });
       res.send(`Task ${taskId} created`);
     } catch (err) {
       require("firebase-functions").logger.error(err);
       res.status(500).send(err.message);
     }
   }
 );
