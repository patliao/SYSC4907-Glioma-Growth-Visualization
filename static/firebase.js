// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import { getFirestore, collection, addDoc, getDocs } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-firestore.js";

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyBjqqP9y8TL-4ehAJRDB7uGeEhQ6bv1HWc",
    authDomain: "glioma-growth-visualization.firebaseapp.com",
    projectId: "glioma-growth-visualization",
    storageBucket: "glioma-growth-visualization.firebasestorage.app",
    messagingSenderId: "943049132912",
    appId: "1:943049132912:web:4263d52960c347b187ddb5",
    measurementId: "G-1S6BKQ1W0X"
  };

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// Test Firebase initialization
console.log("Firebase initialized:", app);

// Function to add data to Firestore
async function addData() {
  try {
    const docRef = await addDoc(collection(db, "gliomaData"), {
      timestamp: new Date(),
      growthRate: 0.5, // Example data
      tumorSize: 10.2, // Example data
    });
    console.log("Document written with ID: ", docRef.id);
  } catch (e) {
    console.error("Error adding document: ", e);
  }
}

// Function to fetch data from Firestore
async function fetchData() {
  try {
    const querySnapshot = await getDocs(collection(db, "gliomaData"));
    querySnapshot.forEach((doc) => {
      console.log(doc.id, " => ", doc.data());
    });
  } catch (e) {
    console.error("Error fetching documents: ", e);
  }
}

// Call addData() and fetchData() after Firebase is initialized
addData().then(() => fetchData());