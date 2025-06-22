import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class NotificationService {
  static final FirebaseMessaging _messaging = FirebaseMessaging.instance;
  static final FlutterLocalNotificationsPlugin _localNotifications = FlutterLocalNotificationsPlugin();

  static Future<void> initialize() async {
    // Request permissions
    await _messaging.requestPermission();

    // Local notification setup
    const AndroidInitializationSettings androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const InitializationSettings initSettings = InitializationSettings(android: androidSettings);
    await _localNotifications.initialize(initSettings);

    // Handle foreground messages
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      RemoteNotification? notification = message.notification;
      AndroidNotification? android = message.notification?.android;
      if (notification != null && android != null) {
        _localNotifications.show(
          notification.hashCode,
          notification.title,
          notification.body,
          const NotificationDetails(
            android: AndroidNotificationDetails(
              'default_channel',
              'Default',
              importance: Importance.max,
              priority: Priority.high,
            ),
          ),
        );
      }
    });
  }

  static Future<void> saveTokenToFirestore() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;
    final token = await _messaging.getToken();
    if (token != null) {
      await FirebaseFirestore.instance.collection('users').doc(user.uid).update({'fcmToken': token});
    }
  }

  static Future<void> subscribeToAnnouncements() async {
    await _messaging.subscribeToTopic('announcements');
  }

  static Future<void> unsubscribeFromAnnouncements() async {
    await _messaging.unsubscribeFromTopic('announcements');
  }
} 