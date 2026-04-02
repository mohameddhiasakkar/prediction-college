import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { ApiService } from './api.service';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should POST to /predict', () => {
    const payload = { country: 'France', skills: 'python', moyenne: '14' };
    service.predict(payload).subscribe((res) => {
      expect(res).toEqual([{ name: 'Test', country: 'France', score: 100 }]);
    });
    const req = httpMock.expectOne('http://127.0.0.1:5000/predict');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(payload);
    req.flush([{ name: 'Test', country: 'France', score: 100 }]);
  });
});
