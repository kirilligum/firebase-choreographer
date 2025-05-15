## Firestore Status Management Conventions for SaaS Applications

When writing Firestore and Firebase Cloud Functions code, adhere to the following conventions for managing procedural statuses:

### 1. Preferred Status Management Approach:

**Use a dedicated status subcollection for complex, multi-step workflows.**  
**Structure**:

```
tasks/{taskId} (document)
  ├── details: { /* main task data */ }
  └── statuses (subcollection)
      └── {statusName}: {
          status: "pending" | "processing" | "completed" | "error",
          updatedAt: Timestamp,
          lastError?: string
      }
```

### Example:

```typescript
// Initiating the first status upon creation
export const initiateTask = functions.firestore
  .document("tasks/{taskId}")
  .onCreate(async (snap) => {
    await snap.ref.collection("statuses").doc("initiated").set({
      status: "completed",
      updatedAt: admin.firestore.FieldValue.serverTimestamp(),
    });
  });
```

### Triggering Subsequent Procedures:

Use Firestore subcollection triggers to handle specific status updates efficiently:

```typescript
export const handleValidatedStatus = functions.firestore
  .document('tasks/{taskId}/statuses/validated')
  .onCreate(async (snap, context) => {
    const taskId = context.params.taskId;
    const taskRef = admin.firestore().collection('tasks').doc(taskId);

    try {
      // Perform next step logic here
      await performNextProcedure(taskId);

      // Mark the next step as completed
      await snap.ref.parent.doc('processed').set({
        status: 'completed',
        updatedAt: admin.firestore.FieldValue.serverTimestamp()
      });
  });
```

### Best Practices:

- **Granular triggers**: Prefer dedicated documents/subcollections for each workflow status.
- **Consistency**: Maintain clear, enumerated statuses (`pending`, `processing`, `completed`, `error`).
- **Error handling**: Clearly store and log errors to facilitate debugging and monitoring.
