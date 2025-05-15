# Get started: write, test, and deploy your first functions

bookmark_border
2nd gen 1st gen

To get started with Cloud Functions, try working through this tutorial, which starts with the required setup tasks and works through creating, testing, and deploying two related functions:

An "add message" function that exposes a URL that accepts a text value and writes it to Cloud Firestore.
A "make uppercase" function that triggers on a Cloud Firestore write and transforms the text to uppercase.
Here's the full sample code containing the functions:

Node.js
Python

// The Cloud Functions for Firebase SDK to create Cloud Functions and triggers.
const {logger} = require("firebase-functions");
const {onRequest} = require("firebase-functions/v2/https");
const {onDocumentCreated} = require("firebase-functions/v2/firestore");

// The Firebase Admin SDK to access Firestore.
const {initializeApp} = require("firebase-admin/app");
const {getFirestore} = require("firebase-admin/firestore");

initializeApp();

// Take the text parameter passed to this HTTP endpoint and insert it into
// Firestore under the path /messages/:documentId/original
exports.addmessage = onRequest(async (req, res) => {
// Grab the text parameter.
const original = req.query.text;
// Push the new message into Firestore using the Firebase Admin SDK.
const writeResult = await getFirestore()
.collection("messages")
.add({original: original});
// Send back a message that we've successfully written the message
res.json({result: `Message with ID: ${writeResult.id} added.`});
});

// Listens for new messages added to /messages/:documentId/original
// and saves an uppercased version of the message
// to /messages/:documentId/uppercase
exports.makeuppercase = onDocumentCreated("/messages/{documentId}", (event) => {
// Grab the current value of what was written to Firestore.
const original = event.data.data().original;

// Access the parameter `{documentId}` with `event.params`
logger.log("Uppercasing", event.params.documentId, original);

const uppercase = original.toUpperCase();

// You must return a Promise when performing
// asynchronous tasks inside a function
// such as writing to Firestore.
// Setting an 'uppercase' field in Firestore document returns a Promise.
return event.data.ref.set({uppercase}, {merge: true});
});

About this tutorial
We've chosen Cloud Firestore and HTTP-triggered functions for this sample in part because these background triggers can be thoroughly tested through the Firebase Local Emulator Suite. This toolset also supports Realtime Database, Cloud Storage, PubSub, Auth, and HTTP callable triggers. Other types of background triggers such as Remote Config and TestLab triggers can be tested interactively using toolsets not described in this page.

Note: You can emulate functions in any Firebase project, but to deploy functions, your project must be on the Blaze pricing plan. See Cloud Functions pricing.
The following sections of this tutorial detail the steps required to build, test, and deploy the sample.

Create a Firebase Project
In the Firebase console, click Add project.

To add Firebase resources to an existing Google Cloud project, enter its project name or select it from the dropdown menu.

To create a new project, enter a project name. You can also optionally edit the project ID displayed below the project name.

Firebase generates a unique ID for your Firebase project based upon the name you give it. If you want to edit this project ID, you must do it now as it cannot be modified after Firebase provisions resources for your project. Visit Understand Firebase Projects to learn about how Firebase uses the project ID.
If prompted, review and accept the Firebase terms.

Click Continue.

(Optional) Set up Google Analytics for your project, which enables an optimal experience using the following Firebase products: Firebase A/B Testing, Cloud Messaging, Crashlytics, In-App Messaging, and Remote Config (including Personalization).

Either select an existing Google Analytics account or create a new account. If you create a new account, select your Analytics reporting location, then accept the data sharing settings and Google Analytics terms for your project.

You can always set up Google Analytics later in the Integrations tab of your settings Project settings.
Click Create project (or Add Firebase, if you're adding Firebase to an existing Google Cloud project).

Firebase automatically provisions resources for your Firebase project. When the process completes, you'll be taken to the overview page for your Firebase project in the Firebase console.

Set up your environment and the Firebase CLI
Node.js
Python
You'll need a Node.js environment to write functions, and you'll need the Firebase CLI to deploy functions to the Cloud Functions runtime. For installing Node.js and npm, Node Version Manager is recommended.

Important: Cloud Functions and the Firebase CLI fully support Node.js versions 18 and 20, with preview-level support for version 22. Versions 14 and 16 are deprecated and will be decommissioned in early 2025 (currently, deployment with these deprecated versions is disabled). See Set runtime options for important information regarding ongoing support for these versions of Node.js.
Once you have Node.js and npm installed, install the Firebase CLI via your preferred method. To install the CLI via npm, use:

npm install -g firebase-tools

This installs the globally available firebase command. If the command fails, you may need to change npm permissions. To update to the latest version of firebase-tools, rerun the same command.

In many cases, new features and bug fixes are available only with the latest version of the Firebase CLI and the firebase-functions SDK. It's a good practice to frequently update both the Firebase CLI and the SDK with these commands inside the functions folder of your Firebase project:

npm install firebase-functions@latest firebase-admin@latest --save
npm install -g firebase-tools
Initialize your project
When you initialize Firebase SDK for Cloud Functions, you create an empty project containing dependencies and some minimal sample code. If you are using Node.js, you can choose either TypeScript or JavaScript for composing functions. For the purposes of this tutorial, you'll also need to initialize Cloud Firestore.

To initialize your project:

Run firebase login to log in via the browser and authenticate the Firebase CLI.
Go to your Firebase project directory.
Run firebase init firestore. For this tutorial, you can accept the default values when prompted for Firestore rules and index files. If you haven't used Cloud Firestore in this project yet, you'll also need to select a starting mode and location for Firestore as described in Get started with Cloud Firestore.
Run firebase init functions. The CLI prompts you to choose an existing codebase or initialize and name a new one. When you're just getting started, a single codebase in the default location is adequate; later, as your implementation expands, you might want to organize functions in codebases.
The CLI gives you these options for language support:

JavaScript
TypeScript
Python
For this tutorial, select JavaScript or Python. For authoring in TypeScript, see Write Functions with TypeScript.

The CLI gives you an option to install dependencies. This is safe to decline if you want to manage dependencies in another way.

After these commands complete successfully, your project structure looks like this:

Node.js
Python

myproject
+- .firebaserc # Hidden file that helps you quickly switch between
| # projects with `firebase use`
|
+- firebase.json # Describes properties for your project
|
+- functions/ # Directory containing all your functions code
|
+- .eslintrc.json # Optional file containing rules for JavaScript linting.
|
+- package.json # npm package file describing your Cloud Functions code
|
+- index.js # Main source file for your Cloud Functions code
|
+- node_modules/ # Directory where your dependencies (declared in # package.json) are installed
For Node.js, the package.json file created during initialization contains an important key: "engines": {"node": "18"}. This specifies your Node.js version for writing and deploying functions. You can select other supported versions.

Import the required modules and initialize an app
After you have completed the setup tasks, you can open the source directory and start adding code as described in the following sections. For this sample, your project must import the Cloud Functions and Admin SDK modules. Add lines like the following to your source file:

Node.js
Python

// The Cloud Functions for Firebase SDK to create Cloud Functions and triggers.
const {logger} = require("firebase-functions");
const {onRequest} = require("firebase-functions/v2/https");
const {onDocumentCreated} = require("firebase-functions/v2/firestore");

// The Firebase Admin SDK to access Firestore.
const {initializeApp} = require("firebase-admin/app");
const {getFirestore} = require("firebase-admin/firestore");

initializeApp();

These lines load the required modules and initialize an admin app instance from which Cloud Firestore changes can be made. Wherever Admin SDK support is available, as it is for FCM, Authentication, and Firebase Realtime Database, it provides a powerful way to integrate Firebase using Cloud Functions.

The Firebase CLI automatically installs the Firebase Admin SDK and Firebase SDK for Cloud Functions modules when you initialize your project. For more information about adding 3rd party libraries to your project, see Handle Dependencies.

Add the "add message" function
For the "add message" function, add these lines to your source file:

Node.js
Python

// Take the text parameter passed to this HTTP endpoint and insert it into
// Firestore under the path /messages/:documentId/original
exports.addmessage = onRequest(async (req, res) => {
// Grab the text parameter.
const original = req.query.text;
// Push the new message into Firestore using the Firebase Admin SDK.
const writeResult = await getFirestore()
.collection("messages")
.add({original: original});
// Send back a message that we've successfully written the message
res.json({result: `Message with ID: ${writeResult.id} added.`});
});

The "add message" function is an HTTP endpoint. Any request to the endpoint results in request and response objects passed to the the request handler for your platform (onRequest() or on_request).

HTTP functions are synchronous (similar to callable functions), so you should send a response as quickly as possible and defer work using Cloud Firestore. The "add message" HTTP function passes a text value to the HTTP endpoint and inserts it into the database under the path /messages/:documentId/original.

Add the "make uppercase" function
For the "make uppercase" function, add these lines to your source file:

Node.js
Python

// Listens for new messages added to /messages/:documentId/original
// and saves an uppercased version of the message
// to /messages/:documentId/uppercase
exports.makeuppercase = onDocumentCreated("/messages/{documentId}", (event) => {
// Grab the current value of what was written to Firestore.
const original = event.data.data().original;

// Access the parameter `{documentId}` with `event.params`
logger.log("Uppercasing", event.params.documentId, original);

const uppercase = original.toUpperCase();

// You must return a Promise when performing
// asynchronous tasks inside a function
// such as writing to Firestore.
// Setting an 'uppercase' field in Firestore document returns a Promise.
return event.data.ref.set({uppercase}, {merge: true});
});

The "make uppercase" function executes when Cloud Firestore is written to, defining the document to listen on. For performance reasons, you should be as specific as possible.

Braces—for example, {documentId}—surround "parameters," wildcards that expose their matched data in the callback. Cloud Firestore triggers the callback whenever new messages are added.

Caution: Be careful to avoid any situation in which the function's result actually retriggers the function, creating an infinite loop — for example, a function triggered by writes to a specific Cloud Firestore document that terminates by writing to that same path. Also make sure to write functions in an idempotent way, so that they produce the same result if they run multiple times for a single event.
In Node.js, event-driven functions such as Cloud Firestore events are asynchronous. The callback function should return either a null, an Object, or a Promise. If you do not return anything, the function times out, signaling an error, and is retried. See Sync, Async, and Promises.

Emulate execution of your functions
The Firebase Local Emulator Suite allows you to build and test apps on your local machine instead of deploying to a Firebase project. Local testing during development is strongly recommended, in part because it lowers the risk from coding errors that could potentially incur cost in a production environment (for example, an infinite loop).

To emulate your functions:

Run firebase emulators:start and check the output for the URL of the Emulator Suite UI. It defaults to localhost:4000, but may be hosted on a different port on your machine. Enter that URL in your browser to open the Emulator Suite UI.

Check the output of the firebase emulators:start command for the URL of the HTTP function. It will look similar to http://localhost:5001/MY_PROJECT/us-central1/addMessage, except that:

MY_PROJECT will be replaced with your project ID.
The port may be different on your local machine.
Add the query string ?text=uppercaseme to the end of the function's URL. This should look something like: http://localhost:5001/MY_PROJECT/us-central1/addMessage?text=uppercaseme. Optionally, you can change the message "uppercaseme" to a custom message.

Create a new message by opening the URL in a new tab in your browser.

View the effects of the functions in the Emulator Suite UI:

In the Logs tab, you should see new logs indicating that your HTTP functions ran successfully:

i functions: Beginning execution of "addMessage"

i functions: Beginning execution of "makeUppercase"

In the Firestore tab, you should see a document containing your original message as well as the uppercased version of your message (if it was originally "uppercaseme", you'll see "UPPERCASEME").

Deploy functions to a production environment
Once your functions are working as desired in the emulator, you can proceed to deploying, testing, and running them in the production environment. Keep in mind that to deploy in production, your project must be on the Blaze pricing plan. See Cloud Functions pricing.

To complete the tutorial, deploy your functions and then execute them.

Run this command to deploy your functions:

firebase deploy --only functions

After you run this command, the Firebase CLI outputs the URL for any HTTP function endpoints. In your terminal, you should see a line like the following:

Function URL (addMessage): https://us-central1-MY_PROJECT.cloudfunctions.net/addMessage
The URL contains your project ID as well as a region for the HTTP function. Though you don't need to worry about it now, some production HTTP functions should specify a location to minimize network latency.

If you encounter access errors such as "Unable to authorize access to project," try checking your project aliasing.

Using the URL output by the CLI, add a text query parameter, and open it in a browser:

https://us-central1-MY_PROJECT.cloudfunctions.net/addMessage?text=uppercasemetoo
The function executes and redirects the browser to the Firebase console at the database location where the text string is stored. This write event triggers the "make uppercase" function, which writes an uppercase version of the string.

After deploying and executing functions, you can view logs in the Google Cloud console. If you need to delete functions in development or production, use the Firebase CLI.

In production, you may want to optimize function performance and control costs by setting minimum and maximum numbers of instances to run. See Control scaling behavior for more information on these runtime options.

# Cloud Firestore triggers

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
