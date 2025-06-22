import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'models/history_entry.dart';

class FirestoreService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final CollectionReference medicalRequests = FirebaseFirestore.instance.collection("medical_requests");
  final CollectionReference history = FirebaseFirestore.instance.collection("history");

  // Add a history entry
  Future<void> addHistoryEntry({
    required String operationType,
    required String description,
    String? question,
    String? answer,
    String? imagePath,
  }) async {
    try {
      final userId = FirebaseAuth.instance.currentUser?.uid;
      if (userId == null) throw Exception("User not authenticated");

      await history.add({
        'timestamp': FieldValue.serverTimestamp(),
        'operationType': operationType,
        'description': description,
        'question': question,
        'answer': answer,
        'imagePath': imagePath,
        'userId': userId,
      });
    } catch (e) {
      debugPrint("Error saving history: $e");
      rethrow;
    }
  }

  // Save a medical request
  Future<void> saveMedicalRequest(
    String imagePath,
    String question,
    String answer,
  ) async {
    try {
      final userId = FirebaseAuth.instance.currentUser?.uid;
      if (userId == null) throw Exception("User not authenticated");

      // Save the medical request
      await medicalRequests.add({
        "imagePath": imagePath,
        "question": question,
        "answer": answer,
        "timestamp": FieldValue.serverTimestamp(),
        "userId": userId
      });

      // Save to history
      await addHistoryEntry(
        operationType: 'medical_request',
        description: 'Medical VQA request processed',
        question: question,
        answer: answer,
        imagePath: imagePath,
      );
    } catch (e) {
      debugPrint("Error saving medical request: $e");
      rethrow;
    }
  }

  // Get history entries
  Stream<List<HistoryEntry>> getHistoryEntries() {
    final userId = FirebaseAuth.instance.currentUser?.uid;
    if (userId == null) return Stream.value([]);

    return history
        .where('userId', isEqualTo: userId)
        .orderBy('timestamp', descending: true)
        .snapshots()
        .map((snapshot) {
          return snapshot.docs
              .map((doc) => HistoryEntry.fromFirestore(doc))
              .toList();
        });
  }

  // Delete a history entry
  Future<void> deleteHistoryEntry(String documentId) async {
    try {
      await history.doc(documentId).delete();
    } catch (e) {
      debugPrint("Error deleting history entry: $e");
      rethrow;
    }
  }

  // Clear all history
  Future<void> clearHistory() async {
    try {
      final userId = FirebaseAuth.instance.currentUser?.uid;
      if (userId == null) throw Exception("User not authenticated");

      final batch = _firestore.batch();
      final snapshot = await history.where('userId', isEqualTo: userId).get();
      
      for (var doc in snapshot.docs) {
        batch.delete(doc.reference);
      }
      
      await batch.commit();
    } catch (e) {
      debugPrint("Error clearing history: $e");
      rethrow;
    }
  }
}