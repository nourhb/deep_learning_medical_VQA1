import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'firestore_service.dart';
import 'models/history_entry.dart';

class HistoryPage extends StatelessWidget {
  final FirestoreService _firestoreService = FirestoreService();

  void _showHistoryDetails(BuildContext context, HistoryEntry entry) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Operation Details'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildDetailRow('Type', entry.operationType),
              _buildDetailRow('Description', entry.description),
              if (entry.question != null) _buildDetailRow('Question', entry.question!),
              if (entry.answer != null) _buildDetailRow('Answer', entry.answer!),
              if (entry.imagePath != null) _buildDetailRow('Image', entry.imagePath!),
              _buildDetailRow('Date', entry.formattedDate),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.grey[700],
            ),
          ),
          SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(fontSize: 16),
          ),
          Divider(),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.history, color: Colors.white, size: 28),
            SizedBox(width: 10),
            Text('History', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          ],
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.delete_sweep),
            onPressed: () async {
              final confirmed = await showDialog<bool>(
                context: context,
                builder: (context) => AlertDialog(
                  title: Text('Clear History'),
                  content: Text('Are you sure you want to clear all history?'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context, false),
                      child: Text('Cancel'),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(context, true),
                      child: Text('Clear'),
                    ),
                  ],
                ),
              );

              if (confirmed == true) {
                await _firestoreService.clearHistory();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('History cleared')),
                );
              }
            },
          ),
        ],
        elevation: 3,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12.0),
        child: StreamBuilder<List<HistoryEntry>>(
          stream: _firestoreService.getHistoryEntries(),
          builder: (context, snapshot) {
            if (snapshot.hasError) {
              return Center(child: Text('Error: ${snapshot.error}'));
            }

            if (snapshot.connectionState == ConnectionState.waiting) {
              return Center(child: CircularProgressIndicator());
            }

            final historyEntries = snapshot.data ?? [];
            if (historyEntries.isEmpty) {
              return Center(child: Text('No history available'));
            }

            return ListView.builder(
              itemCount: historyEntries.length,
              itemBuilder: (context, index) {
                final entry = historyEntries[index];
                return Card(
                  margin: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  elevation: 4,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  child: ListTile(
                    leading: _getOperationIcon(entry.operationType),
                    title: Text(
                      entry.description,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                    subtitle: Text(entry.formattedDate),
                    onTap: () => _showHistoryDetails(context, entry),
                    trailing: IconButton(
                      icon: Icon(Icons.delete, color: Colors.red),
                      onPressed: () async {
                        final confirmed = await showDialog<bool>(
                          context: context,
                          builder: (context) => AlertDialog(
                            title: Text('Delete Entry'),
                            content: Text('Are you sure you want to delete this entry?'),
                            actions: [
                              TextButton(
                                onPressed: () => Navigator.pop(context, false),
                                child: Text('Cancel'),
                              ),
                              TextButton(
                                onPressed: () => Navigator.pop(context, true),
                                child: Text('Delete'),
                              ),
                            ],
                          ),
                        );

                        if (confirmed == true) {
                          await _firestoreService.deleteHistoryEntry(entry.id);
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text('Entry deleted')),
                          );
                        }
                      },
                    ),
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }

  Widget _getOperationIcon(String operationType) {
    IconData iconData;
    Color iconColor;

    switch (operationType) {
      case 'test_image_load':
        iconData = Icons.image;
        iconColor = Colors.blue;
        break;
      case 'image_selection':
        iconData = Icons.photo_library;
        iconColor = Colors.green;
        break;
      case 'medical_request':
        iconData = Icons.medical_services;
        iconColor = Colors.red;
        break;
      case 'error':
        iconData = Icons.error;
        iconColor = Colors.orange;
        break;
      default:
        iconData = Icons.history;
        iconColor = Colors.grey;
    }

    return CircleAvatar(
      backgroundColor: iconColor.withOpacity(0.1),
      child: Icon(iconData, color: iconColor),
    );
  }
}