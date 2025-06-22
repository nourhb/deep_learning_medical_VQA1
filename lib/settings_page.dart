import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firestore_service.dart';
import 'package:provider/provider.dart';
import 'theme_provider.dart';
import 'notification_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class SettingsPage extends StatefulWidget {
  @override
  _SettingsPageState createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  bool isDarkMode = false; // For demo; use Provider or SharedPreferences for persistence
  bool notificationsEnabled = true; // For demo
  final _nameController = TextEditingController();
  final _occupationController = TextEditingController();

  @override
  void dispose() {
    _nameController.dispose();
    _occupationController.dispose();
    super.dispose();
  }

  // Show dialog to edit profile
  void _showEditProfileDialog() async {
    final user = FirebaseAuth.instance.currentUser;
    // Optionally fetch user data from Firestore here
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Edit Profile'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _nameController,
              decoration: InputDecoration(labelText: 'Name'),
            ),
            TextField(
              controller: _occupationController,
              decoration: InputDecoration(labelText: 'Occupation'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              // Save to Firestore (implement as needed)
              // await FirestoreService().updateUserProfile(...)
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Profile updated!')),
              );
            },
            child: Text('Save'),
          ),
        ],
      ),
    );
  }

  // Show dialog to change password
  void _showChangePasswordDialog() {
    final _currentPasswordController = TextEditingController();
    final _newPasswordController = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Change Password'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _currentPasswordController,
              decoration: InputDecoration(labelText: 'Current Password'),
              obscureText: true,
            ),
            TextField(
              controller: _newPasswordController,
              decoration: InputDecoration(labelText: 'New Password'),
              obscureText: true,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              try {
                User? user = FirebaseAuth.instance.currentUser;
                String email = user?.email ?? '';
                AuthCredential credential = EmailAuthProvider.credential(
                  email: email,
                  password: _currentPasswordController.text,
                );
                await user?.reauthenticateWithCredential(credential);
                await user?.updatePassword(_newPasswordController.text);
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Password changed successfully!')),
                );
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Failed to change password: $e')),
                );
              }
            },
            child: Text('Change'),
          ),
        ],
      ),
    );
  }

  // Show dialog to confirm account deletion
  void _showDeleteAccountDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Delete Account'),
        content: Text('Are you sure you want to delete your account? This action cannot be undone.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () async {
              try {
                await FirebaseAuth.instance.currentUser?.delete();
                // Optionally delete user data from Firestore
                Navigator.pushNamedAndRemoveUntil(context, '/log_in', (route) => false);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Account deleted.')),
                );
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Failed to delete account: $e')),
                );
              }
            },
            child: Text('Delete'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Settings')),
      body: ListView(
        children: [
          ListTile(
            leading: Icon(Icons.person),
            title: Text('Edit Profile'),
            onTap: _showEditProfileDialog,
          ),
          ListTile(
            leading: Icon(Icons.lock),
            title: Text('Change Password'),
            onTap: _showChangePasswordDialog,
          ),
          SwitchListTile(
            secondary: Icon(Icons.brightness_6),
            title: Text('Dark Mode'),
            value: context.watch<ThemeProvider>().isDarkMode,
            onChanged: (val) {
              context.read<ThemeProvider>().toggleTheme(val);
            },
          ),
          SwitchListTile(
            secondary: Icon(Icons.notifications),
            title: Text('Enable Notifications'),
            value: notificationsEnabled,
            onChanged: (val) async {
              setState(() => notificationsEnabled = val);
              final user = FirebaseAuth.instance.currentUser;
              if (user != null) {
                await FirebaseFirestore.instance.collection('users').doc(user.uid).update({'notificationsEnabled': val});
              }
              if (val) {
                await NotificationService.subscribeToAnnouncements();
                await NotificationService.saveTokenToFirestore();
              } else {
                await NotificationService.unsubscribeFromAnnouncements();
              }
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.delete_forever, color: Colors.red),
            title: Text('Clear History', style: TextStyle(color: Colors.red)),
            onTap: () async {
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
                await FirestoreService().clearHistory();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('History cleared!')),
                );
              }
            },
          ),
          ListTile(
            leading: Icon(Icons.person_off, color: Colors.red),
            title: Text('Delete Account', style: TextStyle(color: Colors.red)),
            onTap: _showDeleteAccountDialog,
          ),
          ListTile(
            leading: Icon(Icons.logout),
            title: Text('Sign Out'),
            onTap: () async {
              await FirebaseAuth.instance.signOut();
              Navigator.pushReplacementNamed(context, '/log_in');
            },
          ),
        ],
      ),
    );
  }
} 