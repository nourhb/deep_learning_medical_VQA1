import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class SignUpPage extends StatefulWidget {
  @override
  _SignUpPageState createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _nameController = TextEditingController();
  final _surnameController = TextEditingController();

  late String _selectedOccupation;
  late DateTime _selectedDate;
  bool _isLoading = false;

  final List<String> occupations = [
    'Doctor',
    'Radiologist',
    'Medical Student',
    'Nurse',
    'Pharmacist',
    'Patient',
    'Other',
  ];

  @override
  void initState() {
    super.initState();
    _selectedOccupation = occupations.isNotEmpty ? occupations[0] : '';
    _selectedDate = DateTime(1990);
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _nameController.dispose();
    _surnameController.dispose();
    super.dispose();
  }

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
      firstDate: DateTime(1900),
      lastDate: DateTime(2100),
    );
    if (picked != null) {
      setState(() {
        _selectedDate = picked;
      });
    }
  }

  String? validateEmail(String? value) {
    if (value == null || value.isEmpty) return 'Please enter your email';
    final emailRegex = RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$');
    if (!emailRegex.hasMatch(value)) return 'Invalid email address';
    return null;
  }

  String? validatePassword(String? value) {
    if (value == null || value.isEmpty) return 'Please enter a password';
    if (value.length < 6) return 'Password must be at least 6 characters';
    return null;
  }

  String? validatePasswordMatch(String? value) {
    if (value != _passwordController.text) return 'Passwords do not match';
    return null;
  }

  void _submitForm() async {
    if (_formKey.currentState!.validate()) {
      setState(() => _isLoading = true);

      try {
        final String email = _emailController.text.trim();
        final String password = _passwordController.text;
        final String name = _nameController.text;
        final String surname = _surnameController.text;

        final UserCredential userCredential = await FirebaseAuth.instance
            .createUserWithEmailAndPassword(email: email, password: password);

        final User? user = userCredential.user;

        if (user != null) {
          await FirebaseFirestore.instance.collection('users').doc(user.uid).set({
            'name': name,
            'surname': surname,
            'email': email,
            'occupation': _selectedOccupation,
            'birthDate': _selectedDate.toIso8601String(),
          });

          // Send email verification
          await user.sendEmailVerification();

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Registration successful! Please check your email to verify your account.'),
              backgroundColor: Colors.green,
            ),
          );

          Navigator.pop(context); // Go back to login page
        }
      } catch (e) {
        String errorMessage = 'Error during registration. Please check your information.';

        if (e is FirebaseAuthException) {
          switch (e.code) {
            case 'weak-password':
              errorMessage = 'Password must contain at least 6 characters.';
              break;
            case 'email-already-in-use':
              errorMessage = 'This email is already used by another account.';
              break;
            case 'invalid-email':
              errorMessage = 'Invalid email address.';
              break;
            case 'network-request-failed':
              errorMessage = 'No internet connection detected.';
              break;
            default:
              errorMessage = e.message ?? 'Unknown error.';
          }
        }

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(errorMessage), backgroundColor: Colors.red),
        );
      } finally {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).scaffoldBackgroundColor,
              Theme.of(context).scaffoldBackgroundColor,
              Theme.of(context).scaffoldBackgroundColor,
            ],
            stops: [0.0, 0.5, 1.0],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // App Bar personnalisé
              Container(
                padding: EdgeInsets.symmetric(vertical: 20, horizontal: 20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.centerLeft,
                    end: Alignment.centerRight,
                    colors: [
                      Color(0xFF1263AF).withOpacity(0.95),
                      Color(0xFF1263AF).withOpacity(0.85),
                    ],
                  ),
                  borderRadius: BorderRadius.vertical(
                    bottom: Radius.elliptical(MediaQuery.of(context).size.width, 50),
                  ),
                ),
                child: Text(
                  'MedGastro Registration',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    shadows: [
                      Shadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 4,
                        offset: Offset(0, 2),
                      )
                    ],
                  ),
                  textAlign: TextAlign.center,
                ),
              ),

              // Logo avec taille augmentée
              Container(
                margin: EdgeInsets.symmetric(vertical: 30),
                padding: EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Theme.of(context).cardColor,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Theme.of(context).shadowColor.withOpacity(0.1),
                      blurRadius: 15,
                      offset: Offset(0, 5),
                    )
                  ],
                ),
                child: SizedBox(
                  height: 160,
                  child: Image.asset(
                    'assets/images/background.png',
                    fit: BoxFit.contain,
                  ),
                ),
              ),

              // Formulaire
              Expanded(
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 25),
                  decoration: BoxDecoration(
                    color: Theme.of(context).cardColor,
                    borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
                    boxShadow: [
                      BoxShadow(
                        color: Theme.of(context).shadowColor.withOpacity(0.05),
                        blurRadius: 10,
                        offset: Offset(0, -5),
                      )
                    ],
                  ),
                  child: SingleChildScrollView(
                    child: Form(
                      key: _formKey,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.start,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          // Name
                          TextFormField(
                            controller: _nameController,
                            decoration: InputDecoration(
                              labelText: 'Name',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.person, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: (value) {
                              if (value == null || value.isEmpty) return 'Please enter your name';
                              return null;
                            },
                          ),
                          SizedBox(height: 16),

                          // Surname
                          TextFormField(
                            controller: _surnameController,
                            decoration: InputDecoration(
                              labelText: 'Surname',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.person_outline, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: (value) {
                              if (value == null || value.isEmpty) return 'Please enter your surname';
                              return null;
                            },
                          ),
                          SizedBox(height: 16),

                          // Date of Birth
                          TextButton(
                            onPressed: () => _selectDate(context),
                            style: TextButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 16, horizontal: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                                side: BorderSide(color: Color(0xFFB3E5FC)),
                              ),
                            ),
                            child: Align(
                              alignment: Alignment.centerLeft,
                              child: Row(
                                children: [
                                  Icon(Icons.calendar_today, color: Color(0xFF1263AF), size: 20),
                                  SizedBox(width: 10),
                                  Text(
                                    'Date of Birth: ${_selectedDate.toLocal().toString().split(' ')[0]}',
                                    style: TextStyle(
                                      color: _selectedDate.year == 1990
                                          ? Colors.grey
                                          : Colors.black,
                                      fontSize: 16,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          SizedBox(height: 16),

                          // Occupation
                          DropdownButtonFormField<String>(
                            value: _selectedOccupation,
                            items: occupations.map((String value) {
                              return DropdownMenuItem<String>(
                                value: value,
                                child: Text(value),
                              );
                            }).toList(),
                            onChanged: (String? value) {
                              setState(() {
                                _selectedOccupation = value!;
                              });
                            },
                            decoration: InputDecoration(
                              labelText: 'Occupation',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.work, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: (value) {
                              if (value == null || value.isEmpty) return 'Please select your occupation';
                              return null;
                            },
                          ),
                          SizedBox(height: 16),

                          // Email
                          TextFormField(
                            controller: _emailController,
                            keyboardType: TextInputType.emailAddress,
                            decoration: InputDecoration(
                              labelText: 'Email',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.email, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: validateEmail,
                          ),
                          SizedBox(height: 16),

                          // Password
                          TextFormField(
                            controller: _passwordController,
                            obscureText: true,
                            decoration: InputDecoration(
                              labelText: 'Password',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.lock, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: validatePassword,
                          ),
                          SizedBox(height: 16),

                          // Confirm Password
                          TextFormField(
                            controller: _confirmPasswordController,
                            obscureText: true,
                            decoration: InputDecoration(
                              labelText: 'Confirm Password',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.lock_outline, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: validatePasswordMatch,
                          ),
                          SizedBox(height: 24),

                          ElevatedButton(
                            onPressed: _isLoading ? null : _submitForm,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Color(0xFF1263AF),
                              foregroundColor: Colors.white,
                              padding: EdgeInsets.symmetric(vertical: 15),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 5,
                              shadowColor: Color(0xFF1263AF).withOpacity(0.3),
                            ),
                            child: _isLoading
                                ? CircularProgressIndicator(color: Colors.white)
                                : Text(
                              'Register',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                          SizedBox(height: 15),
                          TextButton(
                            onPressed: () {
                              Navigator.pop(context);
                            },
                            child: Text(
                              'Already have an account? Log in',
                              style: TextStyle(
                                color: Color(0xFF1263AF),
                                fontSize: 16,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}