# Firebase Task Manager

A minimal standalone package for managing asynchronous tasks in Cloud Firestore using Firebase Functions.

## Motivation

Firebase Task Manager (FTM) offers several key advantages for building robust and maintainable task-driven workflows:

*   **Persistent State**: Task states are durably stored in Firestore. If a function instance crashes or your system restarts, tasks can resume from where they left off simply by updating their status in Firestore. This ensures that no work is lost and workflows can reliably continue.
*   **Scalable Architecture**: Built on Firestore and Cloud Functions, FTM is designed for horizontal scalability. Firestore handles state management at scale, and Cloud Functions can process a high volume of tasks concurrently, allowing your workflows to grow with demand.
*   **LLM-Friendly "Vibe-Coding"**: The core FTM codebase is tiny (classified as "tiny" in terms of code size that an LLM might need to consider). This small footprint means the essential logic of the library can be easily understood and managed within the context window of modern Large Language Models (LLMs). This makes it exceptionally conducive for AI coding assistants to understand, modify, and generate FTM-compatible code, facilitating a more intuitive "vibe-coding" experience.

### Comparison with Other Workflow Engines

The following table provides a high-level comparison of FTM with other popular workflow orchestration and task management frameworks:

| Framework                        | Persisted Task State                                  | Age (Year) | Examples (small/large) | Code Size        | Deployment           | Linear Scalability          | Robustness                                      | Language | Declaration Style |
|----------------------------------|-------------------------------------------------------|------------|------------------------|------------------|----------------------|-----------------------------|-------------------------------------------------|----------|-------------------|
| Firebase Task Manager (FTM)      | Yes – Firestore docs survive restarts                  | 2023       | small                  | tiny             | serverless           | Yes – Firestore             | Geo-replicated; Functions auto-retry             | JS       | edge              |
| BullMQ (Redis)                   | Yes – job state in Redis persists across restarts      | 2018       | large                  | large            | free self-hosted     | Yes – Redis                 | Redis clustering & master-replica               | JS       | queue             |
| AWS Step Functions               | Yes – SFN service stores step state                    | 2016       | large                  | closed           | serverless           | Yes – SFN service           | Multi-AZ durability & step-by-step replay        | JSON     | graph             |
| GCP Workflows                    | Yes – Workflows service persists each step’s state     | 2021       | large                  | closed           | serverless           | Yes – Workflows service     | Regional durability; optional multi-region      | YAML     | graph             |
| Azure Durable Functions (JS)     | Yes – orchestration state in Azure Storage             | 2018       | large                  | closed           | serverless           | Yes – Azure Storage         | Geo-redundant storage; built-in retries          | JS       | promise           |
| Temporal (TypeScript)            | Yes – history & state in SQL (Postgres/MySQL)          | 2019       | large                  | large            | free self-hosted     | No – SQL (Postgres/MySQL)   | Built-in clustering & multi-DC replication       | TS       | promise           |
| LangGraph (JS/TS)                | Yes – optional Redis or Postgres checkpoint backend   | 2023       | small                  | large            | paid self-hosted     | Yes – Redis; No – Postgres  | Depends on chosen store’s replication setup      | JS/TS    | graph             |
| ReStack (JS/TS)                  | Yes – run state in Postgres or local SQLite            | 2023       | small                  | large            | free self-hosted     | No – SQL (Postgres/SQLite)  | Postgres HA; SQLite single-node only             | JS/TS    | graph             |
| Apache Airflow (Python)          | Yes – DAG state in metadata DB (Postgres/MySQL)        | 2015       | large                  | large            | free self-hosted     | No – SQL metadata DB        | Metadata DB HA; stateless workers                | Python   | graph             |
| llama_index (Python)             | No – pipeline is ephemeral; only index in Vector DB    | 2021       | small                  | large            | free self-hosted     | Yes – Redis                 | Vector DB replication; pipeline ephemeral         | Python   | graph             |
| Haystack (Python)                | No – pipeline in-memory; only docs in ES/FAISS persist | 2019       | small                  | large            | free self-hosted     | N/A                         | No HA, Docker container                          | Python   | edge              |
| AutoGen (Python)                 | No – all state in-memory; lost on restart              | 2022       | small                  | large            | free self-hosted     | No – in-memory              | No built-in replication or failover               | Python   | promise           |

## Prerequisites

- Node.js (>=14.x)  
- Firebase CLI (`npm install -g firebase-tools`)  
- A Firebase project (used by the emulator)

## Installation

```bash
git clone https://github.com/your-org/firebase-task-manager.git
cd firebase-task-manager
npm install
cd functions
npm install
```

## Setup Firebase CLI and Running with Emulator

### Authenticate and Select Project

Ensure you are logged in and have a Firebase project selected:

```bash
firebase login
firebase use --add
```

### Running with Firebase Emulator

Start the Firestore and Functions emulators:

```bash
firebase emulators:start --only firestore,functions
```

Visit the Emulator UI (usually at http://localhost:4000) to monitor Firestore data and function logs.

## Invoking the HTTP Function to Start Task A

Enqueue Task A by sending a JSON body with the `taskId`:

# Remote (deployed) function
```bash
curl -X POST https://us-central1-MY_PROJECT.cloudfunctions.net/startTask \
  -H "Content-Type: application/json" \
  -d '{"taskId":"A"}'
```

# Local (emulator) function
```bash
curl -X POST http://localhost:5001/<your-project-id>/us-central1/startTask \
  -H "Content-Type: application/json" \
  -d '{"taskId":"A"}'
```

You should see emulator logs showing Task A processing, spawning Tasks B & C, and then Task D executing.

## Project Structure

- `functions/firebase-promise` ‑ Core package code  
- `functions/task-watcher.js` ‑ Firestore trigger setup  
- `examples/task-manager-example.js` ‑ Example showing A → (B, C) → D  
- `firestore.rules` ‑ Security rules for the `tasks` collection  
- `firestore.indexes.json` ‑ Index on the `status` field  

## Usage

### Initialization

Import and initialize the Task Manager in your Functions entrypoint:

```js
const admin = require('firebase-admin');
const { logger } = require('firebase-functions');
const { initializeTaskManager, createTaskWatcher } = require('firebase-task-manager');

if (!admin.apps.length) {
  admin.initializeApp();
}

initializeTaskManager(admin, {
  logger,
  tasksPathPattern: 'tasks/{taskId}',
});
```

### Defining Task Handlers

Define your task functions and dependencies:

```js
const userTaskHandlers = {
  A: { handler: taskA, dependencies: [], prefix: false },
  B: { handler: taskB, dependencies: ['^A$'], prefix: false },
  C: { handler: taskC, dependencies: ['^A$'], prefix: false },
  D: { handler: taskD, dependencies: ['^B$', '^C$'], prefix: false },
};
```

### Exporting the Watcher

Export the Firestore trigger:

```js
exports.taskManager = createTaskWatcher(userTaskHandlers, {
  globalFunctionOptions: {
    timeoutSeconds: 120,
    memory: '512MB',
  },
});
```


## API Reference

### initializeTaskManager(adminApp, options)

Initializes the Task Manager with your Firebase Admin SDK and configuration.

- **adminApp**: Firebase Admin instance.  
- **options.logger**: Optional logger (e.g. `functions.logger`).  
- **options.tasksPathPattern**: Firestore path for tasks, must include `{taskId}`.

### createTaskWatcher(userTaskHandlers, options)

Creates a Cloud Function trigger to process tasks.

- **userTaskHandlers**: Object mapping task IDs/prefixes to `{ handler, dependencies, prefix }`.  
- **options.pathPattern**: (Optional) Override default tasks path.  
- **options.globalFunctionOptions**: (Optional) Firebase Functions configuration.

### createChildTasks(event, children)

Enqueues new child tasks within the same collection.

- **event**: Firestore event data.  
- **children**: Array of `{ taskId: string, data: object }`.

### wrapTaskWithOrchestration(handler, dependencies)

(Optional) Advanced utility for composing task functions with dependency checks.

- **handler**: User-defined task function.  
- **dependencies**: Array of regex strings defining dependent task patterns.

## Firestore Security Rules Example

Add the following rules to your `firestore.rules` file to secure the `tasks` collection while allowing your Cloud Functions (using the Admin SDK) to operate unhindered:

```rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow authenticated users to read tasks
    match /tasks/{taskId} {
      allow get, list: if request.auth != null;
      // Admin SDK writes bypass rules, so regular users cannot modify tasks directly
      allow create, update, delete: if false;
    }
    // Deny all other access
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

### Importing the Package in Cloud Functions

```js
const { initializeTaskManager, createTaskWatcher } = require('firebase-task-manager');
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT
