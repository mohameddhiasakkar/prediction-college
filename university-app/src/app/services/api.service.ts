import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = 'http://127.0.0.1:5000/predict';

  constructor(private http: HttpClient) {}

  predict(data: any) {
    return this.http.post(this.apiUrl, data);
  }

  getUniversities(country: string) {
    const q = encodeURIComponent(country);
    return this.http.get(`http://127.0.0.1:5000/universities?country=${q}`);
  }
}