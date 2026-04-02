import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../services/api.service';

@Component({
  selector: 'app-recommendation',
  imports: [CommonModule, FormsModule],
  templateUrl: './recommendation.html',
  styleUrl: './recommendation.css',
})
export class Recommendation {
  student = {
    age: '',
    country: '',
    major: '',
    language: '',
    skills: '',
    moyenne: '',
  };

  results: { name: string; country: string; score: number }[] = [];
  loading = false;
  error: string | null = null;

  constructor(private api: ApiService) {}

  submit() {
    this.error = null;
    this.loading = true;
    this.api.predict(this.student).subscribe({
      next: (res) => {
        this.results = Array.isArray(res) ? res : [];
        this.loading = false;
      },
      error: () => {
        this.results = [];
        this.loading = false;
        this.error =
          'Could not reach the recommendation API. Start the Flask server on port 5000 (python preduction.py from the machine learning folder).';
      },
    });
  }
}
