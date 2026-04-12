/// <reference types="jest" />

import { fetchOperatingPoints } from "./api";
import { apiRequest } from "../../lib/apiClient";

jest.mock("../../lib/apiClient", () => ({
  apiRequest: jest.fn(),
  joinUrl: jest.requireActual("../../lib/apiClient").joinUrl,
}));

describe("fetchOperatingPoints", () => {
  beforeEach(() => {
    const mockedApiRequest = apiRequest as any;
    mockedApiRequest.mockReset();
    mockedApiRequest.mockResolvedValue({ operating_points: [] });
  });

  test("includes dataset_code in the request query", async () => {
    await fetchOperatingPoints("http://localhost:8000", "GCN", "le2i");

    expect(apiRequest).toHaveBeenCalledWith(
      "http://localhost:8000",
      "/api/operating_points?model_code=GCN&dataset_code=le2i"
    );
  });
});
