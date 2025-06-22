import 'package:cloud_firestore/cloud_firestore.dart';

class HistoryEntry {
  final String id;
  final String operationType;
  final String description;
  final DateTime timestamp;
  final String? imageUrl;
  final String? question;
  final String? answer;
  final String userId;
  final String? imagePath;

  HistoryEntry({
    required this.id,
    required this.operationType,
    required this.description,
    required this.timestamp,
    this.imageUrl,
    this.question,
    this.answer,
    required this.userId,
    this.imagePath,
  });

  factory HistoryEntry.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return HistoryEntry(
      id: doc.id,
      operationType: data['operation_type'] ?? '',
      description: data['description'] ?? '',
      timestamp: (data['timestamp'] as Timestamp).toDate(),
      imageUrl: data['image_url'],
      question: data['question'],
      answer: data['answer'],
      userId: data['user_id'] ?? '',
      imagePath: data['imagePath'],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'operation_type': operationType,
      'description': description,
      'timestamp': Timestamp.fromDate(timestamp),
      'image_url': imageUrl,
      'question': question,
      'answer': answer,
      'user_id': userId,
      'imagePath': imagePath,
    };
  }

  String get formattedDate {
    return '${timestamp.day}/${timestamp.month}/${timestamp.year} ${timestamp.hour}:${timestamp.minute}';
  }
} 