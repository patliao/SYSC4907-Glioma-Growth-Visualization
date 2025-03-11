// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-firestore.js";

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