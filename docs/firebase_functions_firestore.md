Firebase
Documentation
Cloud Functions
Build
Was this helpful?

Send feedbackCloud Firestore triggers

bookmark_border
2nd gen 1st gen

With Cloud Functions, you can handle events in Cloud Firestore with no need to update client code. You can make Cloud Firestore changes via the document snapshot interface or via the Admin SDK.

In a typical lifecycle, a Cloud Firestore function does the following:

Waits for changes to a particular document.
Triggers when an event occurs and performs its tasks.
Receives a data object that contains a snapshot of the data stored in the specified document. For write or update events, the data object contains two snapshots that represent the data state before and after the triggering event.
Distance between the location of the Firestore instance and the location of the function can create significant network latency. To optimize performance, consider specifying the function location where applicable.

Cloud Firestore function triggers
The Cloud Functions for Firebase SDK exports the following Cloud Firestore event triggers to let you create handlers tied to specific Cloud Firestore events:

Node.js
Python (preview)
Event Type Trigger
onDocumentCreated Triggered when a document is written to for the first time.
onDocumentUpdated Triggered when a document already exists and has any value changed.
onDocumentDeleted Triggered when a document is deleted.
onDocumentWritten Triggered when onDocumentCreated, onDocumentUpdated or onDocumentDeleted is triggered.
onDocumentCreatedWithAuthContext onDocumentCreated with additional authentication information
onDocumentWrittenWithAuthContext onDocumentWritten with additional authentication information
onDocumentDeletedWithAuthContext onDocumentDeleted with additional authentication information
onDocumentUpdatedWithAuthContext onDocumentUpdated with additional authentication information
Cloud Firestore events trigger only on document changes. An update to a Cloud Firestore document where data is unchanged (a no-op write) does not generate an update or write event. It is not possible to add events to specific fields.

If you don't have a project enabled for Cloud Functions for Firebase yet, then read Get started with Cloud Functions for Firebase (2nd gen) to configure and set up your Cloud Functions for Firebase project.

Writing Cloud Firestore-triggered functions
Define a function trigger
To define a Cloud Firestore trigger, specify a document path and an event type:

Node.js
Python (preview)

import {
onDocumentWritten,
onDocumentCreated,
onDocumentUpdated,
onDocumentDeleted,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.myfunction = onDocumentWritten("my-collection/{docId}", (event) => {
/_ ... _/
});
Document paths can reference either a specific document or a wildcard pattern.

Specify a single document
If you want to trigger an event for any change to a specific document then you can use the following function.

Node.js
Python (preview)

import {
onDocumentWritten,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.myfunction = onDocumentWritten("users/marie", (event) => {
// Your code here
});
Specify a group of documents using wildcards
If you want to attach a trigger to a group of documents, such as any document in a certain collection, then use a {wildcard} in place of the document ID:

Node.js
Python (preview)

import {
onDocumentWritten,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.myfunction = onDocumentWritten("users/{userId}", (event) => {
// If we set `/users/marie` to {name: "Marie"} then
// event.params.userId == "marie"
// ... and ...
// event.data.after.data() == {name: "Marie"}
});
In this example, when any field on any document in users is changed, it matches a wildcard called userId.

If a document in users has subcollections and a field in one of those subcollections' documents is changed, the userId wildcard is not triggered.

Wildcard matches are extracted from the document path and stored into event.params. You may define as many wildcards as you like to substitute explicit collection or document IDs, for example:

Node.js
Python (preview)

import {
onDocumentWritten,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.myfunction = onDocumentWritten("users/{userId}/{messageCollectionId}/{messageId}", (event) => {
// If we set `/users/marie/incoming_messages/134` to {body: "Hello"} then
// event.params.userId == "marie";
// event.params.messageCollectionId == "incoming_messages";
// event.params.messageId == "134";
// ... and ...
// event.data.after.data() == {body: "Hello"}
});
Your trigger must always point to a document, even if you're using a wildcard. For example, users/{userId}/{messageCollectionId} is not valid because {messageCollectionId} is a collection. However, users/{userId}/{messageCollectionId}/{messageId} is valid because {messageId} will always point to a document.

Event Triggers
Trigger a function when a new document is created
You can trigger a function to fire any time a new document is created in a collection. This example function triggers every time a new user profile is added:

Node.js
Python (preview)

import {
onDocumentCreated,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.createuser = onDocumentCreated("users/{userId}", (event) => {
// Get an object representing the document
// e.g. {'name': 'Marie', 'age': 66}
const snapshot = event.data;
if (!snapshot) {
console.log("No data associated with the event");
return;
}
const data = snapshot.data();

    // access a particular field as you would any JS property
    const name = data.name;

    // perform more operations ...

});
For additional authentication information, use onDocumentCreatedWithAuthContext.

Trigger a function when a document is updated
You can also trigger a function to fire when a document is updated. This example function fires if a user changes their profile:

Node.js
Python (preview)

import {
onDocumentUpdated,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.updateuser = onDocumentUpdated("users/{userId}", (event) => {
// Get an object representing the document
// e.g. {'name': 'Marie', 'age': 66}
const newValue = event.data.after.data();

    // access a particular field as you would any JS property
    const name = newValue.name;

    // perform more operations ...

});
For additional authentication information, use onDocumentUpdatedWithAuthContext.

Trigger a function when a document is deleted
You can also trigger a function when a document is deleted. This example function fires when a user deletes their user profile:

Node.js
Python (preview)

import {
onDocumentDeleted,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.deleteuser = onDocumentDeleted("users/{userId}", (event) => {
// Get an object representing the document
// e.g. {'name': 'Marie', 'age': 66}
const snap = event.data;
const data = snap.data();

    // perform more operations ...

});
For additional authentication information, use onDocumentDeletedWithAuthContext.

Trigger a function for all changes to a document
If you don't care about the type of event being fired, you can listen for all changes in a Cloud Firestore document using the "document written" event trigger. This example function fires if a user is created, updated, or deleted:

Node.js
Python (preview)

import {
onDocumentWritten,
Change,
FirestoreEvent
} from "firebase-functions/v2/firestore";

exports.modifyuser = onDocumentWritten("users/{userId}", (event) => {
// Get an object with the current document values.
// If the document does not exist, it was deleted
const document = event.data.after.data();

    // Get an object with the previous document values
    const previousValues =  event.data.before.data();

    // perform more operations ...

});
For additional authentication information, use onDocumentWrittenWithAuthContext.

Reading and Writing Data
When a function is triggered, it provides a snapshot of the data related to the event. You can use this snapshot to read from or write to the document that triggered the event, or use the Firebase Admin SDK to access other parts of your database.

Event Data
Reading Data
When a function is triggered, you might want to get data from a document that was updated, or get the data prior to update. You can get the prior data by using event.data.before, which contains the document snapshot before the update. Similarly, event.data.after contains the document snapshot state after the update.

Node.js
Python (preview)

exports.updateuser2 = onDocumentUpdated("users/{userId}", (event) => {
// Get an object with the current document values.
// If the document does not exist, it was deleted
const newValues = event.data.after.data();

    // Get an object with the previous document values
    const previousValues =  event.data.before.data();

});
You can access properties as you would in any other object. Alternatively, you can use the get function to access specific fields:

Node.js
Python (preview)

// Fetch data using standard accessors
const age = event.data.after.data().age;
const name = event.data.after.data()['name'];

// Fetch data using built in accessor
const experience = event.data.after.data.get('experience');
Writing Data
Each function invocation is associated with a specific document in your Cloud Firestore database. You can access that document in the snapshot returned to your function.

The document reference includes methods like update(), set(), and remove() so you can modify the document that triggered the function.

Node.js
Python (preview)

import { onDocumentUpdated } from "firebase-functions/v2/firestore";

exports.countnamechanges = onDocumentUpdated('users/{userId}', (event) => {
// Retrieve the current and previous value
const data = event.data.after.data();
const previousData = event.data.before.data();

// We'll only update if the name has changed.
// This is crucial to prevent infinite loops.
if (data.name == previousData.name) {
return null;
}

// Retrieve the current count of name changes
let count = data.name_change_count;
if (!count) {
count = 0;
}

// Then return a promise of a set operation to update the count
return data.after.ref.set({
name_change_count: count + 1
}, {merge: true});

});
Warning: Any time you write to the same document that triggered a function, you are at risk of creating an infinite loop. Use caution and ensure that you safely exit the function when no change is needed.
Access user authentication information
If you use one of the of the following event types, you can access user authentication information about the principal that triggered the event. This information is in addition to the information returned in the base event.

Node.js
Python (preview)
onDocumentCreatedWithAuthContext
onDocumentWrittenWithAuthContext
onDocumentDeletedWithAuthContext
onDocumentUpdatedWithAuthContext
For information about the data available in the authentication context, see Auth Context. The following example demonstrates how to retrieve authentication information:

Node.js
Python (preview)

import { onDocumentWrittenWithAuthContext } from "firebase-functions/v2/firestore"

exports.syncUser = onDocumentWrittenWithAuthContext("users/{userId}", (event) => {
const snapshot = event.data.after;
if (!snapshot) {
console.log("No data associated with the event");
return;
}
const data = snapshot.data();

    // retrieve auth context from event
    const { authType, authId } = event;

    let verified = false;
    if (authType === "system") {
      // system-generated users are automatically verified
      verified = true;
    } else if (authType === "unknown" || authType === "unauthenticated") {
      // admin users from a specific domain are verified
      if (authId.endsWith("@example.com")) {
        verified = true;
      }
    }

    return data.after.ref.set({
        created_by: authId,
        verified,
    }, {merge: true});

});
Data outside the trigger event
Cloud Functions execute in a trusted environment. They are authorized as a service account on your project, and you can perform reads and writes using the Firebase Admin SDK:

Node.js
Python (preview)

const { initializeApp } = require('firebase-admin/app');
const { getFirestore, Timestamp, FieldValue } = require('firebase-admin/firestore');

initializeApp();
const db = getFirestore();

exports.writetofirestore = onDocumentWritten("some/doc", (event) => {
db.doc('some/otherdoc').set({ ... });
});

exports.writetofirestore = onDocumentWritten('users/{userId}', (event) => {
db.doc('some/otherdoc').set({
// Update otherdoc
});
});
Note: Reads and writes performed in Cloud Functions are not controlled by your security rules, they can access any part of your database.
Limitations
Note the following limitations for Cloud Firestore triggers for Cloud Functions:

Cloud Functions (1st gen) prerequisites an existing "(default)" database in Firestore native mode. It does not support Cloud Firestore named databases or Datastore mode. Please use Cloud Functions (2nd gen) to configure events in such cases.
Ordering is not guaranteed. Rapid changes can trigger function invocations in an unexpected order.
Events are delivered at least once, but a single event may result in multiple function invocations. Avoid depending on exactly-once mechanics, and write idempotent functions.
Cloud Firestore in Datastore mode requires Cloud Functions (2nd gen). Cloud Functions (1st gen) does not support Datastore mode.
A trigger is associated with a single database. You cannot create a trigger that matches multiple databases.
Deleting a database does not automatically delete any triggers for that database. The trigger stops delivering events but continues to exist until you delete the trigger.
If a matched event exceeds the maximum request size, the event might not be delivered to Cloud Functions (1st gen).
Events not delivered because of request size are logged in platform logs and count towards the log usage for the project.
You can find these logs in the Logs Explorer with the message "Event cannot deliver to Cloud function due to size exceeding the limit for 1st gen..." of error severity. You can find the function name under the functionName field. If the receiveTimestamp field is still within an hour from now, you can infer the actual event content by reading the document in question with a snapshot before and after the timestamp.
To avoid such cadence, you can:
Migrate and upgrade to Cloud Functions (2nd gen)
Downsize the document
Delete the Cloud Functions in question
You can turn off the logging itself using exclusions but note that the offending events will still not be delivered.
