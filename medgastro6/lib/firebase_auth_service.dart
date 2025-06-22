import 'package:firebase_auth/firebase_auth.dart';

class FirebaseAuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;

  // ✅ Retourne un objet Result pour gérer les erreurs sans exceptions
  Future<Result<User?>> signUpWithEmailAndPassword(String email, String password) async {
    try {
      final UserCredential result = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      return Result.success(result.user);
    } on FirebaseAuthException catch (e) {
      return Result.error(_handleAuthException(e, 'sign-up'));
    } catch (e) {
      return Result.error("Unexpected error during sign-up: $e");
    }
  }

  Future<Result<User?>> signInWithEmailAndPassword(String email, String password) async {
    try {
      final UserCredential result = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      return Result.success(result.user);
    } on FirebaseAuthException catch (e) {
      return Result.error(_handleAuthException(e, 'sign-in'));
    } catch (e) {
      return Result.error("Unexpected error during sign-in: $e");
    }
  }

  Future<void> signOut() async {
    try {
      await _auth.signOut();
    } catch (e) {
      rethrow;
    }
  }

  User? getCurrentUser() {
    return _auth.currentUser;
  }

  Future<void> sendEmailVerification() async {
    final user = _auth.currentUser;
    if (user != null && !user.emailVerified) {
      await user.sendEmailVerification();
    }
  }

  Future<bool> isEmailVerified() async {
    final user = _auth.currentUser;
    if (user != null) {
      await user.reload();
      return user.emailVerified;
    }
    return false;
  }

  String _handleAuthException(FirebaseAuthException e, String context) {
    switch (e.code) {
      case 'invalid-email':
        return 'The email address is invalid.';
      case 'user-disabled':
        return 'This user has been disabled.';
      case 'user-not-found':
        return 'No user found for this email.';
      case 'wrong-password':
        return 'Incorrect password.';
      case 'email-already-in-use':
        return 'This email is already in use.';
      case 'operation-not-allowed':
        return 'Email/password accounts are not enabled.';
      case 'weak-password':
        return 'The password is too weak.';
      default:
        return 'An error occurred during $context: ${e.message}';
    }
  }
}

// ✅ Classe utilitaire pour gérer les résultats (succès ou erreur)
class Result<T> {
  final T? value;
  final String? error;

  Result.success(this.value) : error = null;
  Result.error(this.error) : value = null;

  bool get isSuccess => error == null;
}