mod helpers {
    use p3_bn254::Fr;
    use p3_field::{
        PrimeCharacteristicRing, add_scaled_slice_in_place, dot_product, field_to_array,
        par_add_scaled_slice_in_place,
    };

    #[test]
    fn test_add_scaled_slice_in_place() {
        // x = [1, 2], y = [10, 20], scale by 3
        let x1 = Fr::ONE;
        let x2 = Fr::TWO;
        let mut x = vec![x1, x2];
        let mut par_x = x.clone();

        let y1 = Fr::from_u8(10);
        let y2 = Fr::from_u8(20);
        let y = vec![y1, y2];
        let s = Fr::from_u8(3);

        add_scaled_slice_in_place(&mut x, &y, s);
        par_add_scaled_slice_in_place(&mut par_x, &y, s);

        // x = [x1 + s * y1, x2 + s * y2]
        let expected = vec![x1 + s * y1, x2 + s * y2];

        assert_eq!(x, expected);
        assert_eq!(par_x, expected);
    }

    #[test]
    fn test_add_scaled_slice_in_place_zero_scale() {
        let original = vec![Fr::from_u8(4), Fr::from_u8(5)];
        let mut x = original.clone();
        let mut par_x = original.clone();
        let y = vec![Fr::from_u8(6), Fr::from_u8(7)];
        let s = Fr::ZERO;

        add_scaled_slice_in_place(&mut x, &y, s);
        par_add_scaled_slice_in_place(&mut par_x, &y, s);

        assert_eq!(x, original);
        assert_eq!(par_x, original);
    }

    #[test]
    fn test_field_to_array() {
        // Convert value 9 to array of size 4
        let x = Fr::from_u8(9);
        let arr = field_to_array::<Fr, 4>(x);

        // Should yield [9, 0, 0, 0]
        assert_eq!(arr, [x, Fr::ZERO, Fr::ZERO, Fr::ZERO]);
    }

    #[test]
    fn test_field_to_array_single() {
        let x = Fr::from_u8(99);
        let arr = field_to_array::<Fr, 1>(x);
        assert_eq!(arr, [x]);
    }

    #[test]
    fn test_dot_product() {
        let a1 = Fr::TWO;
        let a2 = Fr::from_u8(4);
        let a3 = Fr::from_u8(6);
        let a = [a1, a2, a3];

        let b1 = Fr::from_u8(3);
        let b2 = Fr::from_u8(5);
        let b3 = Fr::from_u8(7);
        let b = [b1, b2, b3];

        // 2*3 + 4*5 + 6*7
        let expected = a1 * b1 + a2 * b2 + a3 * b3;

        let result = dot_product::<Fr, _, _>(a.iter().copied(), b.iter().copied());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<Fr> = vec![];
        let b: Vec<Fr> = vec![];
        let result = dot_product::<Fr, _, _>(a.into_iter(), b.into_iter());
        assert_eq!(result, Fr::ZERO);
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let a1 = Fr::TWO;
        let a2 = Fr::from_u8(4);
        let a = vec![a1, a2];

        let b1 = Fr::from_u8(3);
        let b2 = Fr::from_u8(5);
        let b3 = Fr::from_u8(7);
        let b = vec![b1, b2, b3];

        // Only first two elements will be multiplied
        let expected = a1 * b1 + a2 * b2;

        let result = dot_product::<Fr, _, _>(a.into_iter(), b.into_iter());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_field_to_array_complex() {
        use p3_field::field_to_array;

        // Case 1: Non-zero element, D = 5
        let x = Fr::from_u32(123);
        let arr = field_to_array::<Fr, 5>(x);

        // Should produce: [123, 0, 0, 0, 0]
        assert_eq!(
            arr,
            [Fr::from_u32(123), Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO]
        );

        // Case 2: Zero input value
        let x = Fr::ZERO;
        let arr = field_to_array::<Fr, 3>(x);

        // Should be all zeros: [0, 0, 0]
        assert_eq!(arr, [Fr::ZERO; 3]);
    }
}
