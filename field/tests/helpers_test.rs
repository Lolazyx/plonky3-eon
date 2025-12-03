mod helpers {
    use p3_bn254::Bn254;
    use p3_field::{
        PrimeCharacteristicRing, add_scaled_slice_in_place, dot_product, field_to_array,
        par_add_scaled_slice_in_place,
    };

    #[test]
    fn test_add_scaled_slice_in_place() {
        // x = [1, 2], y = [10, 20], scale by 3
        let x1 = Bn254::ONE;
        let x2 = Bn254::TWO;
        let mut x = vec![x1, x2];
        let mut par_x = x.clone();

        let y1 = Bn254::from_u8(10);
        let y2 = Bn254::from_u8(20);
        let y = vec![y1, y2];
        let s = Bn254::from_u8(3);

        add_scaled_slice_in_place(&mut x, &y, s);
        par_add_scaled_slice_in_place(&mut par_x, &y, s);

        // x = [x1 + s * y1, x2 + s * y2]
        let expected = vec![x1 + s * y1, x2 + s * y2];

        assert_eq!(x, expected);
        assert_eq!(par_x, expected);
    }

    #[test]
    fn test_add_scaled_slice_in_place_zero_scale() {
        let original = vec![Bn254::from_u8(4), Bn254::from_u8(5)];
        let mut x = original.clone();
        let mut par_x = original.clone();
        let y = vec![Bn254::from_u8(6), Bn254::from_u8(7)];
        let s = Bn254::ZERO;

        add_scaled_slice_in_place(&mut x, &y, s);
        par_add_scaled_slice_in_place(&mut par_x, &y, s);

        assert_eq!(x, original);
        assert_eq!(par_x, original);
    }

    #[test]
    fn test_field_to_array() {
        // Convert value 9 to array of size 4
        let x = Bn254::from_u8(9);
        let arr = field_to_array::<Bn254, 4>(x);

        // Should yield [9, 0, 0, 0]
        assert_eq!(arr, [x, Bn254::ZERO, Bn254::ZERO, Bn254::ZERO]);
    }

    #[test]
    fn test_field_to_array_single() {
        let x = Bn254::from_u8(99);
        let arr = field_to_array::<Bn254, 1>(x);
        assert_eq!(arr, [x]);
    }

    #[test]
    fn test_dot_product() {
        let a1 = Bn254::TWO;
        let a2 = Bn254::from_u8(4);
        let a3 = Bn254::from_u8(6);
        let a = [a1, a2, a3];

        let b1 = Bn254::from_u8(3);
        let b2 = Bn254::from_u8(5);
        let b3 = Bn254::from_u8(7);
        let b = [b1, b2, b3];

        // 2*3 + 4*5 + 6*7
        let expected = a1 * b1 + a2 * b2 + a3 * b3;

        let result = dot_product::<Bn254, _, _>(a.iter().copied(), b.iter().copied());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<Bn254> = vec![];
        let b: Vec<Bn254> = vec![];
        let result = dot_product::<Bn254, _, _>(a.into_iter(), b.into_iter());
        assert_eq!(result, Bn254::ZERO);
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let a1 = Bn254::TWO;
        let a2 = Bn254::from_u8(4);
        let a = vec![a1, a2];

        let b1 = Bn254::from_u8(3);
        let b2 = Bn254::from_u8(5);
        let b3 = Bn254::from_u8(7);
        let b = vec![b1, b2, b3];

        // Only first two elements will be multiplied
        let expected = a1 * b1 + a2 * b2;

        let result = dot_product::<Bn254, _, _>(a.into_iter(), b.into_iter());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_field_to_array_complex() {
        use p3_field::field_to_array;

        // Case 1: Non-zero element, D = 5
        let x = Bn254::from_u32(123);
        let arr = field_to_array::<Bn254, 5>(x);

        // Should produce: [123, 0, 0, 0, 0]
        assert_eq!(
            arr,
            [
                Bn254::from_u32(123),
                Bn254::ZERO,
                Bn254::ZERO,
                Bn254::ZERO,
                Bn254::ZERO
            ]
        );

        // Case 2: Zero input value
        let x = Bn254::ZERO;
        let arr = field_to_array::<Bn254, 3>(x);

        // Should be all zeros: [0, 0, 0]
        assert_eq!(arr, [Bn254::ZERO; 3]);
    }
}
