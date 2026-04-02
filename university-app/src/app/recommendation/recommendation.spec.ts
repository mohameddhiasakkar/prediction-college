import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { Recommendation } from './recommendation';

describe('Recommendation', () => {
  let fixture: ComponentFixture<Recommendation>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Recommendation],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(Recommendation);
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('should load recommendations on submit', () => {
    const comp = fixture.componentInstance;
    comp.student.country = 'Greece';
    comp.submit();
    const req = httpMock.expectOne('http://127.0.0.1:5000/predict');
    req.flush([{ name: 'Uni A', country: 'Greece', score: 70 }]);
    expect(comp.results.length).toBe(1);
    expect(comp.results[0].name).toBe('Uni A');
    expect(comp.loading).toBe(false);
  });
});
